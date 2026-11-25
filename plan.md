# plan.md — Gold Factor Research Harness (v4)

> Execution constraints are in `CLAUDE.md` (the project constitution, higher
> priority than this file). This file is only the **phased roadmap and
> acceptance**.

---

## 0. Project claim

> **v4 revision (2026-11-22)**: the claim changed from "sample efficiency" to
> "loop feasibility + same-methodology quality control". The intent is to run the
> agentic workflow end to end and see whether the factors the agent produces are
> in the same class as human brute-force-enumerated ones — **not** to argue lower
> cost. Knock-on effect: **the L1 annotation space and the B3 exhaustive baseline
> are removed entirely**; Phase 0.5's Track A/B fork is void. §0.1 below is the
> current claim, and §0.2 keeps the original v3 claim as a change record.

### 0.1 One sentence (current)

A factor-only **agentic research harness** that answers two things:

1. **Whether this loop runs end to end** — hypothesis → spec → deterministic
   validation → verdict → diagnosis → next round, each link testable,
   reproducible, auditable;
2. **Whether, under the same methodology, the factors the agent produces are in
   the same quality class as the hand-designed L0.**

### Shape of the headline result (use directly when writing the PRD)

> *The harness ran T trials under a fixed budget, accepted K factors, and rejected the rest
> through six deterministic gates. Re-run on the identical train/val split with the identical
> cost model, the agent portfolio's Sharpe is **X** against the hand-designed baseline's **Y**.*

**Report X and Y honestly, whichever is larger.** The agent losing is also a valid
conclusion — it points to whether the bottleneck is the prompt, the search space,
or the gate thresholds. An honest negative result stands up to scrutiny better than
a dressed-up positive one.

⚠️ Since Sharpe is being compared, `CLAUDE.md` A7 is upgraded to a hard
constraint: **the human baseline must re-run the 15 specs in
`harness/l0_catalog.py` on the same train/val split**, never compared against the
README's 0.939.

### 0.2 Original v3 claim (deprecated, kept as a change record)

~~What is measured: under the same trial budget, whether hypothesis-guided search
finds robust factors more efficiently than random search and exhaustion.~~

~~Why "efficiency" rather than "discovery"~~

Discovering new alpha is luck, non-reproducible, and cannot be guaranteed as a
deliverable within a one-quarter practicum. **Efficiency is quantifiable,
falsifiable, and something only you can do** — because only you have (or can build)
a **labeled factor space** as ground truth.

This is exactly RD-Agent(Q)'s own claim: (cite index="23-1">under a cost of under $10, an annualized return about 2× a benchmark factor library while using over 70% fewer factors</cite> — **the value is in "fewer", not "newer".**

~~Headline result: The harness rediscovered K of my N surviving factors within T
trials, versus M evaluations by exhaustive sweep, and beat random search at equal
budget.~~

### Fit to "Taste in the Loop"

The practicum's deliverable is literally **the harness** and an **evolution demo**,
not "a better output". It judges whether **the loop itself works**. So a rigorous
harness is not decoration, it is the product.

| practicum | this project |
|---|---|
| design principles | five-theme economic priors + quant-hygiene rules |
| **critique** | Analysis Unit: gates + structured diagnosis |
| **verify** | Validation Unit: deterministic backtest (no LLM) |
| **revise** | Synthesis Unit: re-propose hypotheses from the diagnosis |
| push past familiar defaults | IC dedup + window robustness, rejecting homogeneous and fragile factors |
| evolution demo | lineage tree + fitness trajectory + rejection-reason distribution + baseline comparison |

---

## 1. Two-layer definition of the factor pool (v4: L1 removed)

| Layer | Content | Role |
|---|---|---|
| **L0** | the 15 hand-designed factors (6 survivors) | **human baseline**, the only control group |
| ~~L1~~ | ~~labeled enumeration space~~ | **removed**. Inventory conclusion: it does not exist, and the new claim does not need it (see §0.1) |
| **L2** | the searchable part of the structure space for the agent | the agent's hunting ground |

**L1 inventory conclusion (Phase 0.5, closed)**: all files under
`final_src_github/results/` — `single_factor_metrics.csv` (45 rows),
`direction_study/single_factor_metrics.csv` (162 rows), `direction_table.csv` (60
rows), `prior_vs_data.csv` (15 rows) — are **all different slicings of those same 15
factors**, with no enumeration-produced labeled cells. Track A is excluded; and
since the new claim does not need a ground-truth space, Track B is void too.

### The agent's task (just one, measurable)

**Within a fixed budget, run the whole loop and produce a set of factors that pass
the six gates; their portfolio is compared against L0's re-run result on the same
split.**

Rediscovering a structure from L0 is an **interesting observation** (it means the
agent's economic priors agree with a human's), but it is **not a success
criterion** — the agent may perfectly well find structures outside L0, as long as
they pass the gates and the comparison result holds up.

### The magnitude of the structure space (why L2 is almost certainly non-empty)

```
13 indicators choose 2         ≈ 156 pairs
× 8 direction operators
× 8 intensity operators
× 5 composition templates
≈ ~tens of thousands (including meaningless combos; the valid subset is much smaller)
```

**And L0 has only 15.** In particular these two kinds are almost certainly
unexplored:
- **cross-theme pairs**: combinations like DXY (theme 1) × SPDR flow (theme 3)
- **same-indicator operator swaps**: GPR used `ma_diff`; has `linreg_slope` /
  `aroon_balance` been tried?

> **Key insight: it does not matter that L1 does not exist — you can build it
> yourself.** With `builder` + `validator`, exhausting a restricted subspace is just
> a `for` loop — tens of minutes of machine time, zero human labor. So the
> "efficiency narrative" holds either way. Phase 0.5 decides which path.

---

## 2. Architecture: five units (modeled on RD-Agent(Q), model branch cut)

RD-Agent(Q) (cite index="27-1">splits quant research into five tightly-coupled units — Specification, Synthesis, Implementation, Validation, Analysis — forming a closed loop under unified input/output constraints that mimics a human researcher's trial-and-error process</cite>. This project is its factor-only simplification.

```
┌──────────────────────────────────────────────────────────────┐
│ ① Specification Unit                     【no LLM · human-fixed】│
│    config.py  : splits / 6 gate thresholds / window grid / MAX_TRIALS │
│    spec.py    : FactorSpec schema + whitelists + canonical hash │
│    duty: guarantee any spec the agent produces can be built and tested │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ ② Synthesis Unit                        【LLM · global view】  │
│    G(H, F): historical hypotheses H + their feedback F → next hypothesis │
│    input:  knowledge/ + trials.jsonl + SOTA set + last diagnosis │
│    output: economic hypothesis + structure choice (theme/pair/operator/template) │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ ③ Implementation Unit                   【no LLM · deterministic】│
│    builder.py: (spec, windows) → pd.Series (reuses your ts_ operators) │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ ④ Validation Unit       【no LLM · deterministic · source of truth】│
│    window-grid scan → sharpe_median / pass_rate / iqr / (max recorded only) │
│    IC dedup (vs SOTA set)                                      │
│    returns train + val only; test physically unreachable       │
└───────────────────────────┬──────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ ⑤ Analysis Unit     【gates no LLM / diagnosis LLM · local view】│
│    6 gates → ACCEPT / REJECT / REFINE                         │
│    diagnosis: translate the numbers into "repair instructions" │
└───────────────────────────┬──────────────────────────────────┘
                            │ write back knowledge/failures + SOTA set
                            └──────────────► back to ②
```

### Four designs inherited directly from RD-Agent

**(1) Local view vs global view separation.** (cite index="27-1">The Analysis Unit works from the local view of the current experiment, the Synthesis Unit keeps a global view across all experiment history, and their interaction balances short-term responsiveness and long-term exploration</cite>. Realized: ⑤'s gates see only the current numbers, ②'s prompt is fed history + diagnosis, and **the two are not mixed into one prompt**.

**(2) Factors are compositional.** RD-Agent's design choice: a factor hypothesis
decomposes into sub-tasks (compute a momentum signal, compute a volatility signal,
then combine), while a model hypothesis maps to a single whole task. **Your
`direction × intensity` template is exactly this decomposition** — no new sub-task
DAG needed; the five composition templates are it.

**(3) IC dedup up front.** (cite index="25-1">Factors are deduped first, computing their correlation against the existing SOTA factor library to filter redundant signals</cite>. The original's threshold is near 1.0 (drop only near-identical); you are single-asset with few factors, so tightened to **0.8**.

**(4) Leakage protection via architecture, not discipline.** RD-Agent's
data-centric design (cite index="22-1">prevents the LLM from accessing raw market data or explicit time splits, mitigating information leakage</cite>. Realized as `CLAUDE.md`'s C2: the agent only gets indicator names and specs, not raw series, let alone the test segment.

### Three deliberate departures from RD-Agent

| RD-Agent | this project | reason |
|---|---|---|
| Co-STEER code-generation agent | spec + deterministic builder | no sandbox, no retry loop; 15h budget |
| joint factor + model optimization | factor-only | your README decision #1: interpretable formulas > ML |
| multi-armed bandit scheduling | LLM decides the next step itself | bandit requires defining the arm set first; goes to BACKLOG |

---

## 3. The source of new factors: flip enumeration from optimization into verification

### The agent owns: three structural dimensions

| Dimension | Space |
|---|---|
| **indicator pairing** | within-theme + cross-theme |
| **operator substitution** | 8 whitelisted operators |
| **composition template** | 5 composition templates |

### The validator owns: the window grid (**not a source of new factors**)

A real factor should hold at both `n=20` and `n=40`; one that holds only at `n=23`
is noise. So the window is not used to optimize but to **verify the structure's
robustness**:

```python
# each window parameter takes {5, 20, 60}; 2 params → 9 cells, 3 params → 27 cells
sharpe_grid = [backtest(build(spec, w)) for w in window_grid]

robustness = {
    "sharpe_median": np.median(sharpe_grid),
    "sharpe_iqr":    iqr(sharpe_grid),
    "pass_rate":     mean([passes_gates(s) for s in sharpe_grid]),
    "sharpe_max":    max(sharpe_grid),   # recorded only, never used in a verdict (CLAUDE.md C5)
}
```

- `pass_rate > 0.4` → the structure holds under most windows → the structure is real
- `pass_rate = 0.07` (2 of 27 cells) with a high `sharpe_max` → **fragile, REJECT**

**Interview framing**: this is **parameter sensitivity analysis**. "I don't pick the
best window; I check whether the structure survives window perturbation."

### FactorSpec (v3, no window parameters)

```json
{
  "spec_version": 3,
  "theme": "risk",
  "template": "direction_x_intensity",
  "direction": {"indicator": "PMF0007", "op": "ma_diff"},
  "intensity": {"indicator": "PMF0042", "op": "abs_pct_change"},
  "hypothesis": "rising geopolitical risk + sharp inflation shifts → rising safe-haven demand",
  "parent_spec_hash": "a3f2...",
  "mutation_note": "swapped the intensity indicator from CPI to TIPS, testing the real-rate channel"
}
```

### Operator whitelist (all reuse your existing ts_ operators)

| op | expansion | # window params |
|---|---|---:|
| `ma_diff` | `sign(ts_mean(x,a) - ts_delay(ts_mean(x,a),b))` | 2 |
| `ma_cross` | `sign(ts_mean(x,a) - ts_mean(x,b))` | 2 |
| `abs_pct_change` | `abs(x / ts_delay(x,n) - 1)` | 1 |
| `zscore` | `(x - ts_mean(x,n)) / ts_std(x,n)` | 1 |
| `linreg_slope` | `ts_linreg_slope(x,n)` | 1 |
| `aroon_balance` | `ts_aroon_balance(x,n)` | 1 |
| `corr_binary` | `ts_corr_binary(x,y,n)` | 1 |
| `rank` | `ts_rank(x,n)` | 1 |

> **Hard rules that must be planted in `knowledge/rules/`**: after a monthly
> indicator is ffill'd to daily, too small a window causes heavy NaN/0 in factor
> values; `ts_mean`'s window must be ≥ `ts_delay`'s step. (Real pitfalls you hit.)

---

## 4. Three benchmark layers

### L1 layer: factor-admission gates (visible to the agent)

```python
GATE_1 = sharpe_median_val > 0
GATE_2 = sharpe_median_val / sharpe_median_train > 0.5      # decay ratio
GATE_3 = pass_rate > 0.4                                    # window robustness
GATE_4 = ann_turnover < TURNOVER_CAP
GATE_5 = max_IC_vs_sota < 0.8                               # signal dedup
GATE_6 = nan_ratio < 0.3

fitness = min(sharpe_median_train, sharpe_median_val)
```

**`min` rather than a weighted average**: a weighting lets a high train score
compensate a low val score, which is exactly what overfitting wants. `min` = count
only the worst segment.

### L2 layer: portfolio reference (deterministic, not optimized)

Uses **vol-normalized CSI 300** (normalized to the portfolio's realized
volatility). README reason #6 holds; unchanged.

Portfolio performance is used only to produce the "which theme to look at next
round" diagnosis, **never as an optimization objective** (`CLAUDE.md` C6).

### L3 layer: system baselines (v4 revision)

| Baseline | How | Answers | Status |
|---|---|---|---|
| **B2 human baseline** | L0's 15 factors, re-run on the **same train/val split + same cost model** | agent output vs human output, same quality class? | **main control** |
| **B1 random search** | same `MAX_TRIALS`, sampling randomly from the structure space, through the identical ④⑤, fixed seed | is the LLM actually doing anything? | **diagnostic control** (not the main result) |
| ~~B3 exhaustion cost~~ | ~~total L1 cells and how many clear~~ | — | **removed** (no L1, and no efficiency claim) |

**B2 is the main control**; it answers the intent directly: how does what this
automated flow makes compare to what human enumeration makes.

**B1 is kept but demoted.** It is cheap (reuses the same ④⑤, only the sampler
changes) and answers an interview must-ask — "how do you know the LLM is doing
something, not getting lucky?". The agent tying random search is **diagnostic
information** (the bottleneck is the prompt or the search space), not a project
failure.

⚠️ **A7 hard constraint**: the control must use the same split, same cost model,
same position mapping. The README's 0.939 cannot be used for comparison.

~~Below is the v3 original:~~

**B1 is the most critical.** If it cannot beat random sampling, the LLM is
decoration. One repo compares exactly this way — (cite index="20-1">two LLMs vs a genetic-programming baseline under the same evaluation budget and 5× the budget</cite>.

⚠️ **Do not take the README's 0.939 as a target** (`CLAUDE.md` A7).

---

## 5. Phased implementation

> Each Phase = a STOP point; its `tests/test_phaseN.py` must pass and the real
> output pasted.
> **The implementation order deliberately puts the agent last**: first a trustworthy
> source of truth (①③④⑤), then introduce the LLM (②). Otherwise there is no way to
> judge whether what the agent produces is right.

---

### Phase -1 — data backfill and indicator inventory · 1.5h

1. **Inventory first**: judge each of the 13 PMF indicators for whether it can be
   extended to 2026. FRED/CFTC-family probably can; Finoview-exclusive, the GPR
   index, and option implied volatility probably cannot. Produce
   `docs/indicator_coverage.md`.
2. Use `akshare` to backfill Au9995 / Shanghai gold + extendable indicators to
   2026-11.
3. Align per the original `align_to_trading_calendar.py` methodology (weekly
   DELAY=3, monthly DELAY=7).
4. Copy external-drive data **to the project-local `data/`** (paths with spaces,
   FAT32, not a runtime dependency).

**`tests/test_phase_minus1.py`**
- sample 3 days of Au9995 close and check they match the official data
- no discontinuity at the join (output a statistics comparison for 2023-10→2024-03)
- `indicator_coverage.md` exists and lists the usable test-segment indicators

**Kill criteria**: if fewer than 5 indicators are extendable, the out-of-time test
segment is insufficient for evaluation → stop and switch to an in-sample split
(train 15–20 / val 21–22 / test 23), and state it in Limitations.

---

### Phase 0 — Specification Unit · 1.5h

1. `config.py`: the three-way split, 6 gate thresholds, window grid `{5,20,60}`,
   `MAX_TRIALS`, `TURNOVER_CAP`, random seed.
2. `spec.py`: schema, whitelist validation, **canonical hash** (sort fields →
   `json.dumps` → `sha256`).
3. `journal.py`: `append_trial` / `check_if_tried(hash)` / `recent_diagnoses(n)` /
   `get_sota_set()`.

**`tests/test_phase0.py`**
- the same spec hashes the same twice; the hash stays the same when field order is
  shuffled
- an illegal operator name / an indicator not in the coverage list → an error with
  a plain-language reason
- after `journal` write it reads back, and `check_if_tried` hits

---

### Phase 0.5 — factor-pool inventory · ✅ **done (2026-11-21) · fork void**

Produced: `docs/pool_inventory.md`. Conclusions:

1. **All 15 L0 factors have spec coordinates**, encoded in
   `harness/l0_catalog.py`, with 15 distinct hashes and full 5-theme 5-template
   coverage.
2. **L1 does not exist**: all files under `results/` are different slicings of those
   15 factors, no enumeration cells.
3. **The Track A/B fork is void**: the v4 claim does not need a ground-truth space
   (see §0.1), and B3 is removed.
4. Side finding: the original plan's FactorSpec v3 could express only 1 of the 15
   L0 → the schema went to v4 (Phase 1's kill criterion triggered and was resolved
   early in Phase 0).

> Below is the v3 original, kept as a change record:

#### ~~Phase 0.5 original design (deprecated)~~ · 1h

**Goal**: decide between Track A and Track B, and fix the project narrative.

Produce `docs/pool_inventory.md`, answering:
1. L0 = which 15 factors, and the coordinates of their structures in the spec space
2. **Whether L1 exists**: are the original enumeration results still around? in what
   form? do they cover the structure dimension or only windows? were cross-theme
   pairs tried?
3. L1 size: how many cells with known indicators

**Fork**

- **Track A — L1 exists and has ≥ 500 labeled cells**: use it directly as the
  ground-truth space. Import it as `data/labeled_space.parquet` (schema:
  `spec_hash, theme, template, ind_a, op_a, ind_b, op_b, sharpe_median, pass_rate,
  verdict`).
- **Track B — L1 missing or too small**: **build it yourself.** After Phase 2, use
  `baselines/exhaustive_sweep.py` to exhaust a **restricted subspace** (suggested: 2
  themes × all indicator pairs × all operators × 2 templates, ~500–2000 cells × 9
  windows ≈ 15–30 minutes of machine time).

> Track B is not a downgrade. A self-built labeled space has a **cleaner
> methodology** (exactly the same validation flow as the harness), and is a more
> credible efficiency yardstick than legacy results.

**`tests/test_phase05.py`**
- `pool_inventory.md` exists, all 15 L0 factors have spec coordinates
- explicitly records which Track was chosen and why

---

### Phase 1 — Implementation Unit · 2h

`builder.py`: `build(spec, windows) -> pd.Series`, importing the existing `ts_*` /
`apply_*`.

**`tests/test_phase1.py` — the single most important acceptance test in the plan**
- hand-write 3 specs + windows, reproducing `pmf_051056` / `pmf_066409` /
  `pmf_055491`
- the produced series equals the corresponding columns of
  `factor_values_top15.csv` **point by point** (`np.allclose`, NaN aligned)

**Kill criteria**: if the 3 factors cannot be expressed as a spec or the numbers do
not match → the spec space does not cover your original designs, **stop and redesign
the schema**, do not proceed.

---

### Phase 2 — Validation Unit · 2.5h

1. `evaluate_series(series, split) -> metrics`, wrapping `position_mapping.py`. Keep
   T+1 lag / 3bp cost / compounded NAV. `split` accepts only train/val (C2).
2. **Window-grid scan** `scan(spec)` → `{sharpe_median, sharpe_iqr, pass_rate,
   sharpe_max, grid_detail}`.
3. **IC dedup**: the IC of the new factor vs the SOTA set, taking the max.

**`tests/test_phase2.py`**
- run existing factors through `evaluate_series`, check against
  `single_factor_metrics.csv`, Sharpe matching to two decimals
- `split="test"` raises
- a synthetic "holds only at n=5" spec → low `pass_rate` but high `sharpe_max`
  (proving the grid distinguishes fragile / robust)
- submit an existing factor as a "new factor" → IC dedup returns near 1.0

~~*(Track B runs exhaustive_sweep.py after this to generate L1.)*~~ Removed in v4.

---

### Phase 3 — Analysis Unit · 2h

1. `analyst.py`: 6 gates → `fitness` → the three-state verdict.
2. **Diagnosis generation**: not `"Sharpe=0.3"`, but `"only 2 of 27 grid cells
   pass, concentrated at the short-window end; train 0.9 / val 0.1, decay ratio
   0.11. The structure fails over long horizons; change the operator rather than the
   window."`
3. REJECT reasons are written back to `knowledge/failures/`.
4. **Threshold calibration**: run the gates over L0's 15 factors. If your
   hand-designed factors are rejected wholesale, the thresholds are too strict →
   adjust them and record the basis in `docs/calibration.md`.

**`tests/test_phase3.py`**
- a high-train / near-0-val spec → GATE_2 catches it and flags overfitting
- a fragile spec → GATE_3 catches it
- a near-copy of an existing factor → GATE_5 catches it
- the reject reason appears in `knowledge/failures/`, and the diagnosis text carries
  an executable suggestion
- paste the gate-pass table for L0's 15 factors

---

### Phase 4 — Synthesis Unit + main loop · 3h

1. `tools.py`: function-calling schemas for the 3 tools
   - `check_if_tried(spec)` → tried or not + historical conclusion
   - `evaluate_spec(spec)` → grid scan + IC dedup (dedups internally first)
   - `write_journal(spec_hash, verdict, note)`
2. `loop.py`: a `while` loop + dispatch. Must have a hard `MAX_TRIALS` cap, a
   per-round tool-call cap, **feeding a validation error back to the agent**, and a
   full trace on disk.
3. **Pre-filter order** (C/A5): spec-hash dedup → complexity → **hypothesis
   alignment** → only then the backtest.
4. **Hypothesis alignment (a first-class deliverable)**: a separate LLM call that
   scores the semantic consistency between `hypothesis` and the spec structure, and
   rejects on mismatch. Stops the agent from inventing an economic story after the
   fact.
   > This is the **one irreplaceable LLM capability in this project**: random search
   > can stumble onto a high Sharpe but cannot give a reason. Your README decision #1
   > wants "a PM able to judge whether the reason for being long still holds this
   > week". **The LLM's value is binding "it works" to "there is a reason".**
5. System prompt: the five-theme framework, the operator whitelist, the indicator
   coverage list, `knowledge/rules/`, and "you cannot see the test set, do not
   request it".

**`tests/test_phase4.py`**
- a small `MAX_TRIALS=3` run; `trials.jsonl` has 3 complete records
- feed an illegal operator name → the agent gets an error and fixes itself
- **assert the round-2 prompt contains the round-1 diagnosis** (proving the feedback
  loop is wired)
- a sample where the hypothesis clearly contradicts the spec → rejected by the
  alignment check

**Kill criteria**: if the agent produces specs that all hit `check_if_tried` for 5
rounds in a row (spinning in place), the global view did not reach the prompt → stop
and fix the prompt, do not paper over it by raising `MAX_TRIALS`.

---

### Phase 5 — same-methodology comparison (**main result**) · 1.5h

1. **B2 (main control)**: L0's 15 specs re-run on the **same train/val split, same
   cost model, same position mapping**, through the identical ④⑤, producing the
   human baseline's factor-layer and portfolio metrics.
2. **B1 (diagnostic control)**: `baselines/random_search.py`, sampling specs at the
   same `MAX_TRIALS`, through the identical ④⑤, fixed seed.
3. **Comparison table**: agent vs human(L0) vs random × {# factors passing the
   gates, median fitness, portfolio Sharpe / ann vol / max drawdown / turnover,
   N_trials, rejection-reason distribution}.
4. **Structural-overlap observation** (not a criterion): does the agent's ACCEPT set
   intersect L0's 15 spec hashes; an intersection means the agent's economic priors
   coincided with a human's, an interesting narrative point.

**`tests/test_phase5.py`**
- the three comparison tables are generated, the numbers come from an actual run,
  and B2's split parameters are identical to the agent's (assert the comparison)
- the random search is reproducible under a fixed seed (two runs give the same
  result)
- assert the report shows **Sharpe and N_trials together** (A9)

**Honesty requirement**: **if the agent cannot beat the human baseline, report it
honestly — that is not a failure.** The claim is "the loop runs + a same-methodology
comparison", not "the agent must win". Where it loses (factor-layer fitness?
portfolio diversification? turnover?) is itself a conclusion.

---

### Phase 6 — evolution report + unlocking test · 2h

1. `make_report.py`: N_trials (prominent), the lineage tree, the fitness trajectory,
   the rejection-reason distribution, the final specs and their economic
   hypotheses, the three-baseline comparison.
2. `unlock_test.py`: evaluate the final factor portfolio on the test segment
   **once**, `--i-understand-this-is-one-shot`, append the result directly to the
   report, and **do not go back and change specs**.
3. **Limitations (must write)**: single-asset, no cross-section; fitting direction on
   train is in-sample; the multiple testing of N_trials trials is not corrected with
   **Deflated Sharpe**; the agent searches only a restricted structure space and this
   is not general factor discovery; some indicators are unavailable in the test
   segment; there is no L1 ground-truth space, so no sample-efficiency claim, only a
   same-methodology comparison.

---

## 6. Budget

| Phase | Content | Duration |
|---|---|---:|
| -1 | data backfill + indicator inventory | 1.5h |
| 0 | Specification Unit | 1.5h |
| 0.5 | factor-pool inventory ✅ done | 1.0h |
| 1 | Implementation Unit | 2.0h |
| 2 | Validation Unit | 2.5h |
| 3 | Analysis Unit + threshold calibration | 2.0h |
| 4 | Synthesis Unit + main loop | 3.0h |
| 5 | system-baseline comparison | 1.5h |
| 6 | report + unlock test | 2.0h |
| | **total** | **17.0h** |

~~Track B adds about 0.5h of human time.~~ v4 removed L1; this item is gone.

---

## 7. Explicitly not doing (→ BACKLOG.md)

LangGraph / any orchestration framework · vector retrieval and embeddings ·
multi-agent split (suggested only after the loop runs) · multi-armed bandit
scheduling · model branch and joint factor-model optimization · free Python codegen
and a sandbox · genetic-algorithm crossover · walk-forward rolling validation ·
strict Deflated Sharpe implementation · Web UI

---

## 8. Still open

1. **Model and cost ceiling** → decides `MAX_TRIALS` (suggested 30–50)
2. **Specific gate-threshold values**: GATE_2's 0.5, GATE_3's 0.4, GATE_5's 0.8 are
   all starting guesses, finalized by Phase 3's L0 calibration
3. ~~**Track A/B** decided by Phase 0.5~~ → closed: L1 does not exist and v4 does not
   need it, the fork is void
