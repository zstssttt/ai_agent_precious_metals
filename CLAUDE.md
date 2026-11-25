# CLAUDE.md — Gold Factor Research Harness

> This file is the project's constitution. It is read at the start of every
> conversation. **When it conflicts with `plan.md`, this file wins.**

## What the project is

An **agentic research harness**: an LLM agent proposes factor **structure
hypotheses** from economic priors → a deterministic engine validates
**robustness** over a window grid → an adjudication layer judges by
anti-overfitting rules and produces a diagnosis → the diagnosis drives the next
round.

**The claim is not "discovering new factors", nor "being cheaper". It is two
things:**

1. **Whether this loop runs end to end** (the deliverable is the harness itself
   + an evolution demo);
2. **Whether, under the same methodology, the factors the agent produces are in
   the same quality class as hand-designed / brute-force-enumerated factors** —
   Sharpe is reported honestly, whoever is higher, and **the agent is not
   presumed to win**.

A factor-only simplification of microsoft/RD-Agent(Q).

> **Claims deliberately not made**: no claim of sample efficiency, no claim of
> lower cost, no claim of discovering new alpha.
> Therefore the **L1 annotation space and the exhaustive baseline (B3) do not
> need to exist at all**.

## Inviolable constraints (breaking one is a serious error)

### C1. Deterministic boundary
The gate halves of `builder.py` / `validator.py` / `analyst.py` **may never call
an LLM**. They are the project's source of truth and must be reproducible. Same
input, same output.

The only places an LLM may be called:
- `loop.py`'s Synthesis (proposing hypotheses, choosing structures)
- `analyst.py`'s **diagnosis-prose generation** (the gate decision itself is
  still `if`)
- hypothesis-alignment scoring (a separate call, input and output both recorded)

### C2. Test set locked
The `split` argument of `evaluate_series(series, split)` **accepts only
`"train"` / `"val"`**. Passing `"test"` must `raise ValueError`.

The test segment is reachable only through `scripts/unlock_test.py`, and that
script requires an explicit `--i-understand-this-is-one-shot`.

**Any implementation that lets the agent see test-segment metrics indirectly is
a bug.** This is an architectural constraint, not a discipline.

### C3. Read-only dependencies
The following files **must not be modified, only imported**:
- `build_factors.py` (8 `ts_` operators + 5 composition templates +
  `PMF_TO_FILENAME`)
- `position_mapping.py` (backtest logic, cost model, Sharpe weighting)
- `direction_study.py` (direction-fitting methodology)

When a change is needed, write a wrapper under `harness/`; do not touch the
original files.

### C4. No filling in financial numbers from memory
Any Sharpe, IC, return, or volatility must be actually computed by code and
backed by evidence. If unsure, run the code — do not estimate.

### C5. The verdict looks at the median, not the maximum
In the window-grid scan results, `sharpe_max` is **recorded only and takes part
in no verdict**. All gates and fitness use only `sharpe_median` and `pass_rate`.

Using `sharpe_max` for selection in the code = parameter-level p-hacking = a
serious error.

### C6. Portfolio Sharpe is the **evaluation endpoint**, not the **search objective**
The factor layer is iterated by the agent; the portfolio layer is the
deterministic Sharpe weighting in `position_mapping.py` — **no LLM, no search,
no tuning**.

The project's final comparison is exactly portfolio Sharpe (agent vs human), but
two things must stay separate:

- **During the search**: the agent cannot see portfolio Sharpe; its objective is
  only the factor-layer `fitness = min(sharpe_median_train, sharpe_median_val)`.
  Portfolio performance feeds back only as "which theme to look at next round".
- **After the search**: portfolio Sharpe is reported as the comparison endpoint.

Wiring portfolio Sharpe into the agent's reward signal = letting it optimize the
final score directly = overfitting; a serious error.

### C7. Failures must leave a trace
Every trial (including validation failures, dedup hits, and gate rejections)
must be written to `journal/trials.jsonl` — not one may be skipped. Failure
records are input to the next round, not garbage.

## Working protocol

1. **One Phase at a time.** Finish, stop, and wait for my confirmation. Do not
   push ahead continuously.
2. **Each Phase must pass its `tests/test_phaseN.py` and paste the real terminal
   output.** Assertions like "done" are not accepted.
3. **Read before you write.** Before touching `build_factors.py`-related logic,
   `view` the actual code to confirm the function signature; do not write from
   memory.
4. **When a feature is optional, default to not doing it**, append it to
   `BACKLOG.md`, and say why it was deferred.
5. **Report the discrepancy rather than hide it.** If an acceptance number does
   not match, say so and by how much — do not tune parameters to make it look
   right.
6. **Phases have kill criteria** (see `plan.md`). When one triggers, stop and
   report; do not route around it.

## Code conventions

- Python 3.10+, `pandas` / `numpy` / `scipy`. No orchestration framework (see A1).
- All paths read from `harness/config.py`, never hardcoded. Data lives in the
  project-local `data/`, with **no dependency on an external drive**.
- One module per unit; modules pass data through explicit data structures
  (`dataclass`), not bare dicts everywhere.
- Randomness must be reproducible: `random_search.py` and any sampling accept a
  `seed`.
- Use `logging`, not `print`.

## Anti-patterns (project-specific; committing one is a directional error)

| Anti-pattern | Why it is wrong |
|---|---|
| A1. Introducing LangGraph / CrewAI / AutoGen | The loop is a sequential cycle, not a graph; framework abstraction makes errors in a 15h project harder to trace |
| A2. Adding vector retrieval / embeddings / FAISS | Already removed explicitly. Memory uses structured queries over `trials.jsonl` |
| A3. Letting the agent generate Python code | The whole value of the spec approach is avoiding the sandbox and traceback-retry loop |
| A4. Putting window parameters in the spec | The spec hash must represent "one structure". Windows belong to the validator's grid |
| A5. Deduping only after the backtest | The backtest is the slowest step. Dedup, complexity, and hypothesis alignment are all **up-front** filters |
| A6. Using `sharpe_max` for any decision | See C5 |
| **A7. Taking the README's 0.939 as a comparison target** (this project compares Sharpe, so this is upgraded to a hard constraint) | 0.939 is an in-sample number: **full-sample direction fit + no train/val split**. The harness uses a train/val split with direction fit on train only. Comparing the two directly is invalid. The human baseline must **re-run** the 15 specs in `harness/l0_catalog.py` under the same split and compare those re-run values |
| A8. Handing the gate decision to an LLM | See C1 |
| A9. Reporting Sharpe without N_trials | The multiple-testing problem must be visible |

## Layout

```
harness/config.py      ① Specification: splits / thresholds / grid / budget
harness/spec.py        ① FactorSpec schema + whitelists + hash
harness/builder.py     ③ Implementation: spec + windows → Series
harness/validator.py   ④ Validation: window grid + IC dedup (deterministic)
harness/analyst.py     ⑤ Analysis: gates + fitness + diagnosis
harness/journal.py     memory: trials.jsonl read/write + dedup queries
harness/tools.py       the 3 tools' schema + dispatch
harness/loop.py        ② Synthesis: the agent main loop
baselines/             random_search.py (B1 diagnostic control; exhaustive_sweep removed)
knowledge/             rules/ · findings/ · failures/
tests/                 test_phase0.py ... test_phase6.py
scripts/               run_research.py · unlock_test.py · make_report.py
```

## Glossary

- **spec**: a factor's **structure** (theme + template + indicator pair +
  operator), with no window parameters
- **pass_rate**: the share of window-grid cells that clear the gates; the
  project's core robustness metric
- **fitness**: `min(sharpe_median_train, sharpe_median_val)`, a conservative
  estimate
- **L0 / L2**: the 15 hand-designed factors (the main control) / the structure
  space the agent can search. ~~L1~~ removed (see `plan.md` §1)
- **SOTA set**: the set of currently-ACCEPTed factors, used for IC dedup and the
  agent's global view
