# MISTAKES.md — bugs hit and how they were fixed

> This file records real bugs and design slips from development — not for
> compliance, but because the failures are themselves a research asset: most of
> them are **silent failures** — the program keeps running, HTTP 200, no
> traceback, the error hiding in the data flow rather than the control flow. In
> an agentic system this class of bug does not crash; it turns into bad input
> fed to the model or a wrong number reported out. They are recorded so a human
> can read them.
>
> Each entry: **symptom → root cause → why it was hard to spot → fix → lesson**.

---

## M1. A reasoning model spends `max_tokens` on thinking and the answer is truncated (silent money leak)

**Symptom**: the trace shows `finish_reason="length"`, `reasoning_tokens == max_tokens`,
and empty text and tool calls. The model thought for 20–70 seconds, was cut off
hard, and emitted not one usable token; the loop had to re-ask once
(`synth.step1`), turning a 2-call round into 3–4 calls.

**Root cause**: in a reasoning model the thinking tokens and the answer tokens
**share one `max_tokens` budget**. With `LLM_MAX_OUTPUT_TOKENS` set to 4096, a
high-effort synthesis call could burn all 4096 on thinking and get cut off
before it ever emitted a tool call.

**Why it was hard to spot**: it **does not raise**. HTTP 200, no exception, and
the retry classifier (correctly) does not treat it as a transport failure. The
only trace is in the single `finish_reason` field. Billing is by generated
tokens, so you pay in full, get zero usable output, and then pay again for the
re-ask.

**Fix**:
1. Root cause — raise `LLM_MAX_OUTPUT_TOKENS` enough to hold "thinking + answer".
   Measured: **flash is fine at 8192, pro needs 16384** (pro thinks far more than
   flash). The ceiling is not the bill: you pay only for tokens actually
   generated, so raising it adds no cost and removes the wasted re-ask.
2. Visibility — `llm.py:complete()` logs a WARNING on `finish_reason == "length"`,
   so this kind of leak can no longer hide in a single field.

**Lesson**: with a reasoning model, `finish_reason=length` + `reasoning≈max_tokens`
is the sneakiest way to burn money — monitor it explicitly.

---

## M2. A verdict that contradicts its own diagnosis (a consistency invariant across two functions)

**Symptom**: trial 7's verdict was `REFINE` ("worth mutating further"), but the
same record's diagnosis read "the structure is invalid, do not keep mutating
it". That diagnosis goes **verbatim into the next round's prompt**, handing the
agent two opposite instructions at once: "mutate it" and "do not mutate it".

**Root cause**: the two code paths keyed off **different criteria**. `decide()`
looks at `pass_rate` and the **val** Sharpe; `_suggestions()` looks at the
**train** Sharpe. For a "train≤0 but val>0" factor — a loss in sample, a fluke
positive out of sample — they necessarily disagree: decide sees val positive and
says REFINE, suggestions sees train negative and says "invalid".

**Why it was hard to spot**: `decide()` and `_suggestions()` **each had unit
tests, and both passed**. The granularity of a unit test is "one function",
while the constraint "the verdict and the diagnosis must say the same thing"
**belongs to no single function — it only exists between the two**. This is a
**consistency invariant**, which unit tests inherently cannot catch. It was
found by reading the journal, not by a test.

**Fix**: the fix was to the verdict, not the prose. `decide()` now recognizes
the signal — `pass_rate == 0` and a non-positive median in **either** split →
REJECT. The basis is the project's own anti-overfitting principle: a structure
that clears zero cells and also loses money in sample has no room to mutate, and
that small positive on val is a fluke on a short validation window. After the
fix the verdict and the diagnosis agree. (`_suggestions()`'s instinct was right;
`decide()` had the gap.)

**Lesson**: when two functions' outputs are consumed together, write a
**contract test** asserting their consistency — do not just test each in
isolation.

---

## M3. GATE_5 signal dedup silently disabled (a serialization round-trip asymmetry)

**Symptom**: from the first ACCEPT onward, the 40-trial production run logged
`WARNING  SOTA record ... no longer parses` every round. Once an accepted factor
was written to the journal it could never be read back. All three ACCEPTs had a
GATE_5 observed |IC| of `0.000`.

**Root cause**: a **serialization asymmetry**. The spec the agent emits has
`legs` as a **list**; `spec_from_payload` parses it into a `FactorSpec` (whose
`legs` is a **dict**), and the journal stores the dict form. But on read-back
`sota_series()` called `spec_from_payload` again, which only accepts the list →
it raised `SpecError` on the dict form → the SOTA series dict stayed empty →
`max_ic_vs_sota` always compared against an empty set → always returned 0.0 →
**GATE_5 was inert for the whole run, never actually deduping anything**.

**Why it was hard to spot**: it logged a WARNING but **did not stop the run**;
the run finished and reported "3 ACCEPTs" as if dedup had worked. A textbook case
of "a successful return ≠ the right thing was done".

**Impact assessment (verified afterwards)**: rebuilding the three factors by hand
and computing pairwise Spearman |IC| gave a maximum of only **0.278** (well below
the 0.70 threshold) — so this run's result happened **by luck** not to be
corrupted; the three ACCEPTs really are mutually distinct. But that was luck, not
the gate working.

**Fix**: make `spec_from_payload` **accept both the dict and list forms of
`legs`** (convert a dict to a list first, then run the original logic), so the
read/write round trip is symmetric and backward-compatible with existing
journals. Plus a round-trip regression test: *any spec serialized into the
journal must be readable back by spec_from_payload*.

**Lesson**: any data that is "written out and read back" needs a round-trip
test; otherwise the two ends' formats drift apart silently, raising nothing, and
failing silently downstream.

---

## M4. Upstream brotli decoding raises TypeError after the 200 (an environment-dependent false success)

**Symptom**: DeepSeek returns `Content-Encoding: br`, and httpx2's BrotliDecoder
on anaconda's bundled `brotli` package raises `TypeError: process() takes no
keyword arguments` **after a 200 was received** — the request succeeded but the
decode crashed.

**Root cause**: `httpx2._decoders` calls a C function (positional-args only) with
a keyword argument; the `brotli` and `brotlicffi` variants have incompatible
APIs, and anaconda installed the broken one.

**Fix**: pin `Accept-Encoding` to `gzip` in `LLMClient`, bypassing brotli. At
this payload size gzip costs nothing. Remove once httpx2 is fixed or
`brotlicffi` is the installed variant.

**Lesson**: an exception after a 200 is the easiest to misread as "a bug in our
code". It is often an environment/dependency pitfall — first check whether the
exception is raised after the request succeeded or before.

---

## Appendix: phenomena filed as "normal", not bugs

- **~20–35s slow first run**: not a background process, but Python cold-start
  import (pandas/numexpr) plus a cold OS file cache. Near-zero CPU time with high
  wall-clock time = blocking wait. The second run is back to sub-second. Not a
  code issue; not fixed.
- **Low ACCEPT rate**: under strict gates even the human factors only partly
  pass; a low pass rate is partly by design (anti-overfitting), not a factor-
  quality bug.
