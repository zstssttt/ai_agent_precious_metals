# BACKLOG.md

> CLAUDE.md working protocol #4: an optional feature defaults to not-done, is
> appended here, with a note on why it was deferred.
> Ordering = recommended handling order, not importance.

---

## Found by Phase 0, to be decided in a later Phase

### B1. CSI 300 benchmark data missing — Phase 2 will hit this
`position_mapping.py:88` wants `000300_SH.csv`, and `补齐后数据/` has no CSI 300
series at all (searched the whole directory with `find`).
Limited blast radius: the benchmark is used only for the portfolio-layer control
(plan.md §4 L2); **the factor-layer gates and fitness do not need it at all**.
Deferral reason: Phase 2 only does `evaluate_series`, which does not use the
benchmark. Phase 5's portfolio control must resolve it; akshare's
`stock_zh_index_daily` can backfill it directly (a China source, no proxy needed).

### B2. The indicator whitelist took only 13 of 42 available series
`补齐后数据/` holds 26 daily + 10 monthly + 6 weekly. Ready-made series outside the
whitelist include: VIX, gold/silver ratio, Fed total assets, COMEX gold
speculative net longs/longs, US M2 YoY, University of Michigan consumer
expectations, total foreign holdings of US Treasuries, bank reserve balances, SOFR
3M, 30Y breakeven, US gold-ETF volatility index, fed funds rate, Shanghai gold
close/settlement, Au(T+D), and 4 non-SPDR ETFs' monthly flow and cash flow.
**Expanding the whitelist would multiply L2 and is the highest-ROI add**, but each
one first needs: a theme assignment + a real coverage-period check (ffill fake-tail
problem, see B3) + a correlation de-redundancy against existing indicators.
Deferral reason: 13 indicators are enough to run the loop; expanding the whitelist
would add irrelevant variables to Phase 1's L0 reproduction test. Suggested after
Phase 2 runs and before Phase 4.

### B3. Data only reaches 2025-11-28, about a year ago
Real observations uniformly stop at 2025-11-28 (GPR to 2026-02-13 is an exception).
Today is 2026-11-21.
Blocks no Phase: the test segment `2024-01-01 → 2025-11-28` is already genuinely
out-of-time, about 2 years.
Deferral reason: backfilling to 2026-11 needs akshare (Au9995/Shanghai gold, a
China source, directly reachable) + FRED (UST/TIPS/CPI/spread, needs a proxy) + the
GPR site, while COMEX, option IV, SPDR flow, and the onshore-offshore spread are
vendor-exclusive and probably cannot be backfilled → it would leave indicators with
ragged coverage periods and actually contaminate the split. Benefit below cost.

### ~~B4. Track A/B decision~~ — **closed (2026-11-22), not doing it**
The project's claim changed to "loop feasibility + same-methodology quality
control", and no longer claims sample efficiency.
→ The L1 ground-truth space and the B3 exhaustive baseline are removed entirely;
`exhaustive_sweep.py` is not written.
→ The measured 561-million space size is downgraded to background context
(supporting "the agent searches in a very large space"), no longer an exhaustion-
feasibility argument.

### B5. The pre-transform cap of 2 is a guess
`config.MAX_PRE_TRANSFORMS_PER_INPUT = 2` exactly fits L0's most complex
`pct_ret+abs`. No evidence that 3 would be better or worse.
Deferral reason: loosening grows the space by another order of magnitude, and L0
does not need it. Wait for Phase 4 to see whether the agent is actually limited by
this.

### ~~B6. The gate thresholds are all starting guesses~~ — **calibrated in Phase 3, see `docs/calibration.md`**
Conclusion: five of the six kept their original value; only **GATE_5 tightened from
0.8 to 0.70** was recommended (not yet applied, pending confirmation).
The live-run verdict distribution ACCEPT 3 / REFINE 10 / REJECT 2; plan.md's
presumed "rejected wholesale → loosen" trigger did not hold.
The original record below is kept as the basis.

### B6-original record
`GATE_2=0.5` / `GATE_3=0.4` / `GATE_5=0.8` come from plan.md's suggested values;
only `GATE_4=25` has a data basis (L0 measured turnover 7.77–20.67).

**Phase 2 live run (full-grid scan of 15 L0 specs, 75 cells, train/val split):**

| Gate | current threshold | L0 pass count |
|---|---|---|
| GATE_1 `sharpe_median_val > 0` | 0.0 | **9 / 15** |
| GATE_3 `pass_rate > 0.4` | 0.4 | **3 / 15** |
| GATE_4 `turnover < 25` | 25.0 | **15 / 15** (measured median 7.56–16.73) |
| GATE_6 `nan_ratio < 0.3` | 0.3 | **12 / 15** |
| `fitness > 0` | — | **7 / 15** |

GATE_3 lets through only 3 — exactly the "hand-designed factors rejected
wholesale" situation plan.md Phase 3 anticipated. But **do not loosen the threshold
on that basis**; first separate two possibilities:
(a) the threshold is too strict;
(b) these 15 L0 factors were selected under a **full-sample in-sample
methodology**, and should drop out once switched to a train/val split (which is
exactly what A7 is about).

Evidence favors (b): `pmf_051056` has full-sample Sharpe 0.466 and made the
portfolio, but in the grid `sharpe_max=0.753` while `sharpe_median_val=-0.038`,
`pass_rate=0.22` — a textbook fragile case.
GATE_4 is the only 15/15 pass, meaning the cap of 25 is too loose; Phase 3 could
consider tightening to 20.

Deferral reason: calibration belongs to Phase 3, to be done together with the
diagnosis prose and `docs/calibration.md`. Phase 2 only computes the numbers.

### B9. `scan()`'s cell count is between 3 and 27, not a fixed 27
Of L0's 15 specs, 9 have only 1 window role (3 cells) and 6 have 2–3 roles (9
cells). `pass_rate` on 3 cells has a resolution of only {0, ⅓, ⅔, 1}, not the same
precision as a 27-cell pass_rate.
Impact: GATE_3's 0.4 threshold is effectively "pass at least 2 cells (0.667)" for a
3-cell spec and "pass at least 11 cells" for a 27-cell spec — **the same threshold
is stricter or looser depending on spec complexity**.
Deferral reason: either set thresholds tiered by cell count, or add a penalty in
`fitness` for low-resolution specs — both are Phase 3 adjudication-layer design, and
need to first see the agent's actual spec-complexity distribution.

### B7. `MAX_TRIALS = 40` not tied to cost

### B8. The factor series contain inf, and the published results are affected too — found in Phase 1
A full-grid scan of 15 L0 specs over 75 cells produced **972 inf** total:

| factor | whole-series inf | in-window inf | inf in published CSV | inf date range |
|---|---|---|---|---|
| `pmf_128453` ★survivor | 67 | 0 | 67 | 2005-03-28 ~ 2012-01-30 |
| `pmf_125861` ★survivor | 67 | **2** | 67 | 2005-03-24 ~ 2017-12-08 |
| `pmf_127111` | 54 | **2** | 54 | 2005-03-24 ~ 2017-12-08 |
| `pmf_102135` | 258 / 255 / 271 (three grid cells) | — | — | — |

Two things:
1. **Handled on the harness side**: `nan_ratio()` counts with `np.isfinite`, so inf
   counts as unusable just like NaN, and GATE_6 catches it. One of `pmf_102135`'s
   cells is 12% inf in-window; counting NaN only would pass it.
2. **Not handled upstream**: `results/factor_values_top15.csv` carries these infs,
   meaning they flowed into the backtest at the time. The cause is a path not
   covered by `zero2nan` in the `ratio_over_logsize` / `corr_change` family.

Deferral reason: C3 says `build_factors.py` is read-only; we cannot fix it there.
There are only 2 in-window points (2/2189 = 0.1%), a negligible effect on the B2
human baseline's Sharpe. A real fix would add a `sanitize()` wrapper in `harness/` —
but that would break Phase 1's point-by-point reproduction test (the harness and
published values would no longer match). **Suggested: when Phase 5 runs the B2
baseline, state it honestly in the report's Limitations and do not change the
numbers.**

### B7. `MAX_TRIALS = 40` not tied to cost — **closed 2026-11-24**
Measured: `deepseek-v4-pro`, 40 trials = 117 calls / $0.827 / 91 minutes, 91.9%
prompt-cache hit. ~$0.021 amortized per trial. Under a $20 budget, 40 trials is 4%;
cost is not a constraint, the threshold stays 40.

### B10. `journal.read_trials` silently skips corrupted lines
A line that cannot be constructed into a `TrialRecord` only logs a WARNING and
`continue`s. One corrupted ACCEPT record vanishes straight out of the SOTA set,
**silently weakening GATE_5**, and does not trigger the new `raise` added in
`sota_series()` (see README §6). Found 2026-11-24 while writing the GATE_5
regression test — my initial test corrupted a line-level field, the record was
skipped in `read_trials`, and never reached the code under test.

Deferral reason: the correct fix (strict mode / checksum / refuse to start) touches
the journal's read/write contract, and the journal is the carrier of C7. Do not
change it right after a run just finished.
**Any claim that a run's GATE_5 was valid must first confirm that run's journal has
no skipped lines.**

### B11. `llm_trace.jsonl` lacks `trial_id`
The trace records the full prompt, tokens, elapsed time, and finish_reason, but has
no join key to tie back to a round — only timestamp order. 40 trials produce 117
records, and aggregate analysis (e.g. "which round was slowest, and why") is
impossible.

Deferral reason: passing round context to `LLMClient.complete()` changes Phase 4's
core call path. **The cost is asymmetric**: adding a field at write time is nearly
free, but it can never be recovered after the fact — must be changed before the next
40-trial run.

### B12. The agent gets REFINE but does not mutate
Of the 40 trials, 13 were judged REFINE (the engine explicitly says "worth mutating
further"), but only 3 actually filled in `parent_spec_hash` (trials 12 / 19 / 23).
Behaviorally this is closer to random restart than evolution, directly affecting
whether the project can be called an "evolution demo".

Deferral reason: this is a prompt / tool-design issue, not a bug. The fix candidates
(force the next round after a REFINE to carry a parent, or backfill the last round's
spec as a tool default) all change the agent's decision surface, so they **must be
done as a separate experiment arm, not mixed into an existing run**.

### B13. round_prompt's failure histogram is binned by "reason", not by "indicator"
Consequence: the feedback loop is blind along the indicator dimension. Over 40
trials PMF0042 had zero coverage and PMF0031 only 1 use, yet those two are the most
frequently used in the human pool (7 each). The agent gets no "you haven't tried X"
information.

Deferral reason: this is the second experiment arm after B12. **Run the as-is arm A
first**; only an A/B difference can answer "is the dimensionality of the feedback
signal the bottleneck". Adding it now would contaminate the control.

---

## Explicitly not doing per plan.md §7 (on record, not up for discussion again)

| Item | Reason not to do it |
|---|---|
| LangGraph / CrewAI / AutoGen and other orchestration frameworks | A1: the loop is a sequential cycle, not a graph |
| vector retrieval / embeddings / FAISS | A2: memory uses structured queries over `trials.jsonl` |
| multi-agent split | consider only after the single-agent loop runs end to end |
| multi-armed bandit scheduling | requires defining the arm set first, extra engineering; the LLM deciding autonomously is itself the selling point |
| model branch / joint factor-model optimization | factor-only is a deliberate departure (README decision #1) |
| free Python codegen + sandbox | A3: the whole value of the spec approach is avoiding the sandbox and retries |
| genetic-algorithm crossover | goes to BACKLOG |
| walk-forward rolling validation | the fixed three-way split is enough |
| strict Deflated Sharpe implementation | just state honestly in Limitations that no haircut was applied |
| Web UI | the deliverable is the harness, not a product |
