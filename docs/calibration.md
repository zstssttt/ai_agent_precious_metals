# docs/calibration.md — threshold calibration of the six gates

> plan.md Phase 3 item 4: *run the gates over L0's 15 factors. If your
> hand-designed factors are rejected wholesale, the thresholds are too strict →
> adjust them and record the basis in this file.*
>
> Generated 2026-11-22, **changes applied 2026-11-23**. Data methodology:
> `train 2015-01-01→2021-12-31` / `val 2022-01-01→2023-12-31`, direction fit on
> train only, position mapping `rolling_percentage`, all actually run by
> `harness/validator.scan()`.
>
> **Applied: `GATE_3` 0.4 → 0.3, `GATE_5` 0.8 → 0.70.** The other four unchanged.
> Decision maker: the project owner; §3.3 of this file originally recommended
> keeping GATE_3 at 0.4, and was overridden — see §3.3-resolution.

---

## 1. Live run results

```
factor      verdict      fit  GATE_1 GATE_2 GATE_3 GATE_4 GATE_5 GATE_6
pmf_003678  ACCEPT     0.002   o      o      o      o      o      o
pmf_011095  REFINE     0.092   o      X      X      o      o      o
pmf_051056  REFINE    -0.038   X      X      X      o      o      o
pmf_051113  REFINE    -0.168   X      X      X      o      o      o
pmf_055491  REFINE     0.032   o      o      X      o      o      o
pmf_058341  REFINE    -0.052   o      X      X      o      o      o
pmf_066409  REFINE     0.159   o      o      X      o      o      o
pmf_069545  REFINE     0.000   X      X      X      o      o      X
pmf_102135  REJECT     0.000   X      X      X      o      o      X
pmf_104203  REFINE    -0.569   X      X      X      o      o      o
pmf_118259  REJECT    -0.239   X      X      X      o      o      X
pmf_125861  REFINE    -0.049   o      X      X      o      o      o
pmf_127111  ACCEPT     0.211   o      o      o      o      o      o
pmf_128453  ACCEPT     0.259   o      o      o      o      o      o
pmf_131810  REFINE     0.130   o      o      X      o      o      o

pass counts: GATE_1 9/15, GATE_2 6/15, GATE_3 3/15, GATE_4 15/15, GATE_5 15/15, GATE_6 12/15
verdicts: ACCEPT 3, REFINE 10, REJECT 2
```

The GATE_5 column was run with an empty SOTA set, so it was effectively not
tested. See the separate experiment in §3.5.

---

## 2. Were they "rejected wholesale"? — No

The verdict distribution is **ACCEPT 3 / REFINE 10 / REJECT 2**.

REFINE is not a rejection; it is "this structure is worth mutating further, and
the diagnosis says exactly how". Only 2 were sentenced to death, and both were
monthly CPI factors caught by GATE_6 — they are **already NaN** in the original
project's `single_factor_metrics.csv`, not a loss newly caused by this project.

So plan.md's presumed trigger "rejected wholesale → loosen thresholds" **does not
hold**, and no loosening is performed.

---

## 3. Gate by gate

### 3.1 GATE_1 `sharpe_median_val > 0` — 9/15 pass · **keep**

A little over half pass, a healthy resolution. No change needed.

### 3.2 GATE_2 `val/train > 0.5` — 6/15 pass · **keep**

Of the 9 failures, **5 are because `train ≤ 0`** (the decay ratio is undefined and
fails by design), not because of real decay.

This design is worth stating plainly: a factor that loses in sample and makes
money out of sample has no "decay" to measure. Handing it a flattering
negative-over-negative ratio would let luck clear an anti-overfitting gate. The
same rule runs at the cell layer (`validator._cell_verdict`) and the spec layer
(`analyst.decay_ratio`), so the two cannot disagree.

> If this is ever loosened, the right move is not to lower 0.5 but to add a
> "train must be positive" pre-gate, splitting "overfitting" and "never worked"
> into two distinct diagnoses. Merging them now is a deliberate simplification.

### 3.3 GATE_3 — **changed to 0.3** (this section originally recommended keeping 0.4, overridden)

#### 3.3-resolution (2026-11-23)

`GATE_3_MIN_PASS_RATE = 0.4 → 0.3`. Effect: GATE_3 pass count 3/15 → 12/15, the
verdict distribution ACCEPT 3 / REFINE 10 / REJECT 2 → **ACCEPT 6 / REFINE 6 /
REJECT 3**, and the SOTA set 3 → 6. Four factors flip: `pmf_066409` `pmf_131810`
`pmf_055491` → ACCEPT; `pmf_011095` → REJECT (once the SOTA set grows, its maxIC
0.796 hits the tightened GATE_5).

The reasoning for this decision: 3 SOTA members are too thin for Phase 4 — if
GATE_5's dedup baseline has only 3 factors, that gate barely engages, amounting to
a gate set for nothing. 6 is the minimum size that lets dedup actually work.

**The cost that must be stated in the report**: of `pass_rate`'s 15 values, **9
are exactly 0.333**, with no sample between 0.3 and 0.4. So this is not a tweak —
it is a cliff that flips 60% of the sample at once. Of those 9, **6 are "1 of 3
cells"** — under 0.3, "only one of three windows worked" counts as window-robust.
That is in tension with what the name GATE_3 is meant to express, and we must not
pretend otherwise.

> The real fix is resolution-aware thresholds (BACKLOG B9), not moving a single
> global number. A third option — keep 0.4 but require a spec to have at least 2
> window roles (≥9 cells) — was not adopted, because it would shut out all 9
> single-role structures in L0.

#### Below is the original analysis (recommended keeping 0.4, not adopted)

Originally 3/15 pass. Three reasons not to loosen:

This is the only genuinely tight gate, and the one most easily misjudged as
"thresholds too strict". Three reasons not to loosen:

**(a) These 15 L0 factors were selected under a full-sample in-sample methodology.**
Dropping out under a train/val split is the **expected result, not the gate's
fault** (CLAUDE.md A7). The hardest evidence is `pmf_051056`: full-sample Sharpe
0.466, made the final portfolio, but under the grid scan `sharpe_max=0.753` while
`sharpe_median_val=-0.038`, `pass_rate=0.22` — one cell dazzling, eight no good, a
textbook parameter accident. This is exactly why GATE_3 exists.

**(b) Lowering a threshold to let one's own baseline pass is precisely the move
this project sets out to avoid.** The B2 human baseline is a **control group**,
not a **pass mark**. A control group scoring badly is a conclusion, not a fault.

**(c) The resolution problem is real, but the fix is not lowering the threshold.**
9 of the 15 specs have only 1 window role (3 cells), so `pass_rate` can only take
{0, 0.33, 0.67, 1}. For a 3-cell spec, 0.4 is effectively "pass at least 2/3"; for
a 9-cell spec it is "pass at least 4/9" — the same threshold is stricter or looser
depending on spec complexity. This is **BACKLOG B9**, to be solved with tiered
thresholds or a fitness penalty; lowering one global number only smears the
problem over rather than solving it.

### 3.4 GATE_4 `turnover < 25` — 15/15 pass · **keep, but filed as non-binding**

Measured median turnover ranges **7.56 – 16.73**, and the cap of 25 leaves 8.3 of
headroom above the maximum. This gate **does not engage at all** on L0.

Reason not to tighten: there is no evidence that turnover 17 is worse than turnover
10 — the cost model is a linear 3bp, already deducted in the Sharpe, so adding a
hard cap on top is double punishment. The role of 25 is to catch pathological
high-frequency churn (e.g. an operator that flips direction every day), which would
be far above 25.

> `tests/test_phase3.py::test_l0_gate_table` has an assertion pinning
> `GATE_4 == 15/15`. The day an agent-produced factor makes it start engaging, that
> assertion goes red, prompting a revisit of this section.

### 3.5 GATE_5 — **tightened from 0.8 to 0.70**

With an empty SOTA set this gate always passes, so it was not tested. A sequential
experiment was added: submit the 15 factors in descending fitness order, ACCEPTs
join the SOTA set, and later entrants compute `max |Spearman IC|` against the
existing SOTA.

Distribution of all 105 pairwise |IC| in L0:

| statistic | value |
|---|---|
| median | 0.033 |
| p90 | 0.117 |
| max | **0.796** |

The top four pairs:

| \|IC\| | pair | why it is a near-duplicate |
|---|---|---|
| **0.796** | `pmf_011095` × `pmf_128453` | both built on the correlation structure of `diff+rank(PMF0012)` |
| **0.723** | `pmf_051056` × `pmf_051113` | **identical direction leg** (`ma_diff(PMF0007)`), only the intensity changed |
| 0.414 | `pmf_125861` × `pmf_128453` | same template, same risk leg, one indicator changed |
| 0.310 | `pmf_011095` × `pmf_125861` | — |

Two things:

1. **A threshold of 0.8 catches none of them**, yet the first two pairs are visibly
   duplicates — the second pair's direction leg is character-for-character identical.
2. There is a clear gap in the distribution between **0.723 and 0.414**. Any
   threshold in (0.42, 0.72] catches exactly those two pairs and misjudges none of
   the other 103.

**Applied: `GATE_5_MAX_IC_VS_SOTA = 0.70`.**
Taking 0.70 rather than 0.5 hugs the upper edge of the gap and stays conservative;
this is not fitting a single point, it is reading a bimodal distribution.

This change is **coupled** with the GATE_3 change: after the SOTA set grows from 3
to 6, later entrants' `maxIC` baseline rises overall (measured 0.011→0.030,
0.047→0.116, 0.054→0.140), so `pmf_011095` turns from REFINE into REJECT. The two
changes cannot be evaluated separately.

### 3.6 GATE_6 `nan_ratio < 0.3` — 12/15 pass · **keep**

Catches 3: `pmf_069545` (0.309), `pmf_118259` (0.310), `pmf_102135` (0.889).

All are the R1 pitfall: monthly CPI ffill'd to daily and bound to a short window.
**Independent cross-check** — these 3 factors are already NaN in the original
project's `single_factor_metrics.csv`, so the gate and the historical results point
at the same batch of factors.

The first two sit at 0.309 / 0.310, barely over 0.3. We do not loosen for them:
their window binding already violates R1 (`pmf_069545`'s `abs_zscore(PMF0042)` uses
window 5), so being caught is correct behavior.

---

## 4. Conclusion

| Gate | old | new | action |
|---|---|---|---|
| GATE_1 | 0.0 | 0.0 | keep |
| GATE_2 | 0.5 | 0.5 | keep |
| GATE_3 | 0.4 | **0.30** | **loosen** (§3.3-resolution; cost filed) |
| GATE_4 | 25.0 | 25.0 | keep, filed as non-binding |
| GATE_5 | 0.8 | **0.70** | **tighten** (§3.5) |
| GATE_6 | 0.3 | 0.3 | keep |

L0 live run after the change:

```
pass per gate  GATE_1 9/15  GATE_2 6/15  GATE_3 12/15  GATE_4 15/15  GATE_5 15/15  GATE_6 12/15
verdicts       ACCEPT 6 / REFINE 7 / REJECT 2      (empty-SOTA methodology, GATE_5 always passes)
verdicts       ACCEPT 6 / REFINE 6 / REJECT 3      (sequential-SOTA methodology, GATE_5 actually engages)
```

`tests/test_phase3.py::test_calibrated_thresholds_match_the_calibration_doc` pins
these six numbers. Change a threshold without updating here and the test goes red.

## 5. When to come back and revisit this

- the agent goes 10 rounds in a row in Phase 4 without a single ACCEPT → GATE_3 may
  really be too tight; check whether `failures_by_reason()`'s histogram is dominated
  by GATE_3;
- the `GATE_4 == 15/15` assertion in `test_l0_gate_table` goes red → turnover is
  starting to engage;
- after the whitelist expands to 42 indicators (BACKLOG B2) → GATE_5's IC
  distribution will change and the gap may shift;
- any time someone wants to "loosen a threshold because the baseline did not pass" →
  re-read §3.3(b) first.
