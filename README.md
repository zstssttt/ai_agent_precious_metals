# Gold-Timing Factors · Agentic Research Harness

A research loop that **proposes, validates, and iteratively improves quantitative factors on its own**. An LLM agent treats a factor as an **economic hypothesis** expressed as a structured spec; a deterministic engine builds and backtests it over a window grid; an anti-overfitting adjudication layer judges it **ACCEPT / REFINE / REJECT** and writes a diagnosis; the diagnosis feeds the next round. It runs on real precious-metals daily data.

This is a **factor-only simplification** of Microsoft's **RD-Agent(Q)** idea: separate **research** (hypothesis → structure, done by the LLM) from **development** (build → backtest → adjudicate, done by deterministic code), then close the loop so each round learns from the last one's failures.

> **Scope statement**: this project claims only two things — (1) the loop runs end to end; (2) whether, under the same methodology, agent factors are in the same quality class as hand-designed factors. It does not claim to discover new alpha, to be cheaper, or to be sample-efficient. Sharpe is reported honestly, whoever is higher.
>
> The repo is built on top of a **read-only** upstream gold-factor project (`final_src_github/`): the upstream backtest, position-mapping, and direction-fitting code is unchanged, imported only through the single boundary `harness/legacy.py`.

---

## 1. Architecture: the life cycle of a spec

```
 ┌─ Round N ───────────────────────────────────────────────────┐
 │                                                             │
 │  ② loop.py: round_prompt() builds the global view          │  ← how it returns for round 2:
 │              (SOTA set / recent diagnoses)                  │    last round's diagnosis is folded
 │        │                                                    │    by round_prompt into this prompt
 │        ▼                                                    │
 │  ★ agent emits evaluate_spec(spec = JSON)  [only LLM call]  │  ← where the Agent is called: only here
 │        │                                                    │  ← how the Spec is written: the agent
 │        ▼                                                    │    emits JSON per tools.py's schema
 │  tools.spec_from_payload(JSON) → FactorSpec  [parse/check]  │    (theme/template/legs/hypothesis)
 │        │                                                    │
 │        ▼  —— wall: below here the agent does not participate (C1) —— │
 │  evaluate_spec orchestrates three read-only scripts:        │
 │     scan()   validator.py  window-grid backtest (calls build repeatedly)  │  ← where Implementation is:
 │     build()  builder.py    spec + windows → pandas.Series   │    builder.build, called by scan
 │     judge()  analyst.py    six gates → Verdict (+ diagnosis)│
 │        │                                                    │
 │        ▼                                                    │
 │  journal.append_trial() → trials.jsonl (one line per trial) │  ← how the Trial JSON is updated:
 │        │                                                    │    _record_for builds a TrialRecord,
 └────────┼────────────────────────────────────────────────────┘    appended as one line
          ▼
     next round: diagnosis + SOTA set fed back to ②
```

| Question | Answer | Location |
|---|---|---|
| Where is the Agent called? | Only in ② Synthesis, emitting an `evaluate_spec` tool call | `loop.py` `_one_trial` |
| Where is Implementation? | `builder.build`, called by `validator.scan` for every grid cell | `builder.py:256` |
| How does it return for round 2? | Last round's diagnosis is folded by `round_prompt` into the next prompt | `prompts.py:142` |
| How is the Trial JSON updated? | Each round `_record_for` builds a `TrialRecord`, and `append_trial` appends one line | `loop.py:400` |
| How is the Spec written? | The agent emits `theme/template/legs/hypothesis` as JSON per `tools.py`'s JSON Schema | `tools.py:125` |

**Deterministic boundary (C1)**: the gate halves of `builder` / `validator` / `analyst` never call an LLM; same input, same output. The agent can only influence *which structure to propose*, never *how it is scored* — this is an architectural anti-cheating guarantee, not a discipline. The LLM appears in only three places: proposing hypotheses, generating the diagnosis prose, and hypothesis-alignment scoring.

---

## 2. How anti-overfitting is done

This is the real content of the project. Each mechanism blocks a specific way to cheat:

| Mechanism | The cheat it blocks |
|---|---|
| **Window parameters are not in the spec**. The agent only names roles `w1/w2/w3`; the engine sweeps every combination over `{5,20,60}` | "a factor tuned into existence". The windows are already fully swept, so there is nothing left to tune |
| **The verdict uses `pass_rate` and medians; `sharpe_max` is recorded only and takes part in no decision** | picking the single best cell as the conclusion. 1 of 9 cells holding is a parameter accident, not an effect |
| **Direction is fit on the train segment only** | fitting direction on the full sample (the upstream default; see A7) |
| **The test segment is architecturally unreachable**. `evaluate_series(series, "test")` raises `ValueError` outright | looking at the test set repeatedly. This is an architectural constraint, not a discipline |
| **Hypothesis-alignment scoring runs before the backtest** | slapping together a structure then inventing an economic story. A score < 3 is MISALIGNED and the backtest never runs |

### The six gates

```
GATE_1  val-segment median Sharpe > 0.0
GATE_2  decay ratio val/train > 0.5          (a non-positive train fails outright)
GATE_3  pass_rate > 0.3                       (share of grid cells that clear the gates)
GATE_4  median annualized turnover < 25.0
GATE_5  max |IC| vs already-accepted factors < 0.70
GATE_6  share of unusable values in the evaluation window < 0.30

search objective  fitness = min(train median Sharpe, val median Sharpe)   conservative
```

The threshold calibration is justified in [`docs/calibration.md`](docs/calibration.md).

---

## 3. Results: one 40-trial run (deepseek-v4-pro)

```
model   deepseek-v4-pro (strict function calling, /beta endpoint)
scale   40 trials · 117 LLM calls · $0.83 · 91 minutes · 92% prompt-cache hit
verdicts  REJECT 19 · REFINE 13 · ACCEPT 3 · MISALIGNED 4 · DUPLICATE 1
```

### Accepting a factor is a Sharpe test

The gates are essentially Sharpe tests: `GATE_1` requires val-segment median Sharpe > 0, `GATE_2` requires a decay ratio val/train > 0.5, and `GATE_3` requires that Sharpe to be robust across the `{5,20,60}` window grid (pass_rate > 0.3). Splits: train 2015–2021, val 2022–2023; the test segment is never touched (C2).

### The three factors that cleared all six gates

| trial | theme | structure | val Sharpe | train | decay ratio | pass_rate |
|---|---|---|---:|---:|---:|---:|
| 25 | rates | term spread PMF0013: trend × change z-score | +0.330 | +0.383 | 0.86 | 67% |
| 19 | rates | TIPS real yield PMF0002: trend × deviation intensity | +0.240 | +0.091 | — | 44% |
| 18 | etf | ETF holdings PMF0019: path efficiency × aroon trend | +0.082 | +0.057 | 1.44 | 33% |

The three factors are mutually distinct (pairwise signal |IC| measured at 0.096 / 0.238 / 0.278, all well below GATE_5's 0.70), spanning three different channels: real rate, term spread, and ETF flow. trial 25 has the most balanced train/val (0.383 / 0.330, decay ratio 0.86) and clears 6 of its 9 cells.

### Human baseline (reference, not a same-methodology contest)

The 15 hand-designed baseline factors (`harness/l0_catalog.py`) are the **survivors the upstream project brute-force-enumerated from tens of thousands of candidates**, re-scored here by the same engine under the **same split and same gates** (`scripts/baseline_report.py`; A7: do not compare against the upstream README's full-sample 0.939). Six still pass under this methodology, with the following val Sharpe:

```
pmf_066409 (risk)       +0.606
pmf_127111 (goldprice)  +0.555
pmf_128453 (dollar)     +0.522
pmf_055491 (risk)       +0.298
pmf_131810 (etf)        +0.268
pmf_003678 (rates)      +0.158
```

These two sets are **not a same-methodology comparison**: the human set is the product of tens of thousands of enumerations, while the agent's three come from a single 40-configuration iteration. Placing them side by side only gives a **reference range** — the human survivors' val Sharpe lands in 0.16–0.61, the agent's three land in 0.08–0.33, and the best (trial 25, 0.330) reaches into the lower half of the human range.

> Methodology note: the val window is only 2 years (2022–2023), so a raw val Sharpe can be amplified by those two years' market regime — this holds for both sides, and is why the search uses the window-grid median rather than the single-cell peak.

---

## 4. Observed limitations (measured on this run)

- **The agent barely mutates; it is doing random restarts.** Across 13 REFINE verdicts (the engine explicitly says "this structure is worth mutating further"), only 3 rounds actually filled in `parent_spec_hash` (trials 12, 19, 23). As it stands, this is closer to **LLM-guided random search with anti-overfitting gates** than to real evolution. Fixing it needs prompt / tool-design changes, not more budget.
- **Indicator coverage has blind spots.** Of the 13 whitelisted indicators, the agent touched only 9 across 40 trials; PMF0001 / PMF0011 / PMF0036 / PMF0042 were never used. `round_prompt`'s failure histogram is binned by **reason**, not by **indicator**, so the feedback loop is blind along the indicator dimension.
- **Grid resolution is inconsistent.** A single-window-role spec has only 3 cells, so `pass_rate` can only take `{0, 0.33, 0.67, 1.0}`; a GATE_3 threshold of 0.3 means "1 of 3 cells" clears the robustness check, whereas a 9-cell spec needs more than 3. The same gate is stricter or looser depending on spec complexity.

---

## 5. How to run

```bash
pip install openai pandas numpy scipy
export DEEPSEEK_API_KEY='your-key-here'          # full setup in docs/setup_api.md

python3 scripts/run_research.py --dry-run                    # free, prints the prompt
python3 scripts/run_research.py --smoke                      # 3 rounds, cheap model
python3 scripts/run_research.py --trials 40 --model deepseek-v4-pro
python3 scripts/run_research.py --status                     # journal summary

python3 scripts/baseline_report.py                           # re-run the human baseline, same methodology
python3 scripts/view.py trials                               # human-readable trial cards
python3 scripts/view.py trace --finish length                # find truncated calls
python3 -m pytest tests/ -q                                  # full test suite, no network
```

**Resumable**: if it crashes mid-run, re-run the same command; it continues from `journal.next_trial_id()`. To start over, **move** `journal/trials.jsonl` aside (do not delete it — it is the experiment record).

---

## 6. Mistakes made during development

Real bugs and design slips (token exhaustion / truncation, a verdict contradicting its own diagnosis, GATE_5 silently disabled, etc.) are recorded separately in [`docs/MISTAKES.md`](docs/MISTAKES.md) — most of them are **silent failures** (HTTP 200, no traceback, the error hiding in the data flow), and they are this project's most valuable engineering lessons.

---

## 7. Not yet done

| | Item | Why it matters |
|---|---|---|
| B1 | Portfolio-level comparison (the agent's 3 vs the human 6, under deterministic Sharpe weighting) | Portfolio Sharpe is the **evaluation endpoint** (C6). Blocked on missing CSI 300 benchmark data |
| — | Make the agent actually mutate along REFINE | Currently only 3 of 13 REFINEs were followed up. This is the key to upgrading "random restart" into "evolution" |
| — | Bin `round_prompt`'s feedback by **indicator** | Currently blind along the indicator dimension; the agent left 4 indicators untouched over 40 trials |
| — | Add `trial_id` to `llm_trace.jsonl` | Currently call records cannot be tied back to a round |

Full backlog in [`BACKLOG.md`](BACKLOG.md); project constraints in [`CLAUDE.md`](CLAUDE.md).

---

## Layout

```
harness/config.py      ① splits / thresholds / grid / budget
harness/spec.py        ① FactorSpec schema + whitelists + hash
harness/builder.py     ③ spec + windows → Series
harness/validator.py   ④ window grid + IC dedup (deterministic)
harness/analyst.py     ⑤ gates + fitness + diagnosis
harness/llm.py         provider adapter (the only module that touches a model)
harness/tools.py       the 3 tools' schema + dispatch
harness/loop.py        ② main loop + hypothesis alignment + resume
harness/journal.py     memory: trials.jsonl read/write + dedup queries
harness/legacy.py      the single import boundary onto read-only upstream code (C3)
scripts/               run_research.py · baseline_report.py · l0_report.py · view.py
tests/                 test_phase0.py … test_phase4.py
journal/               trials.jsonl · llm_trace.jsonl
docs/                  setup_api.md · calibration.md · MISTAKES.md
```
