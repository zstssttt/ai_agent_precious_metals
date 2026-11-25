# rules/ — verified hard rules (must obey)

> The first of the knowledge base's three buckets. `rules` = verified, stable
> rules (must obey); `findings` = empirical findings (reference); `failures` =
> disproved paths (do not repeat). Accumulated across sessions — not chat
> history, but a structured research asset.
>
> This file goes verbatim into the Synthesis system prompt. Changing it changes
> the agent's behavioral boundary.

---

## R1. After a monthly indicator is ffill'd to daily, the window must not be too small

`PMF0018` (option IV), `PMF0031` / `PMF0036` (SPDR monthly flow and cash flow),
and `PMF0042` (core CPI) are natively monthly, and are ffill'd onto the trading
calendar by `align_to_trading_calendar.py` after a 7-day publication lag.

Consequence: **the same value repeats for about 20 trading days**. So

- taking `diff` / `pct_ret` of the ffill'd series yields 0 on 19 days of each month;
- `ts_std` is identically 0 on a small window → any operator that divides by std
  (`abs_zscore`, `boll_position`) yields NaN or inf;
- `ts_corr_binary` degenerates to constant-vs-constant on a small window, and the
  correlation is undefined.

**Rule**: when a monthly indicator feeds `abs_zscore` / `boll_position` /
`corr_*`, its window role should bind to `w2` (20) or `w3` (60), not `w1` (5).
This is a pitfall the project actually hit — it is the root cause of the heavy
NaN/0 in factor values.

> Note L0 itself violates this: `pmf_069545`'s `abs_zscore(PMF0042)` uses window
> 5. This is exactly why the window-grid scan exists — it tells us whether that
> cell is propped up by noise.

## R2. `ts_mean`'s window must be ≥ `ts_delay`'s step

`ma_diff` expands to `sign(ts_mean(x,a) - ts_delay(ts_mean(x,a),b))`. When
`b > a` it compares a "just-smoothed value" against a "much older smoothed
value" — that is not a trend judgement, it is measuring drift.

**Rule**: bind `ma_diff`'s two slots to the same role (the default), or ensure
the second slot is no longer than the first.

## R3. Denominators must pass through `zero2nan`

Every division in `build_factors.py` is wrapped in `zero2nan`. Any new operator
must do the same, or inf will contaminate the whole series.

GATE_6's `nan_ratio()` now counts **all non-finite values** (NaN and ±inf), so
inf is caught — but that is the last net, not an excuse to skip this rule: being
caught means the whole spec is void, not that those few points are repaired.
Measured: across the full 75-cell L0 grid, 972 infs were produced, and
`pmf_102135` had one cell with 12% inf in the window.

## R4. Direction is fixed on the train segment; no look-ahead

`direction_study.py`'s methodology: a factor's direction is set by the sign of
its IC. That sign **may be computed only on the train segment**. Fixing
direction on the full sample is in-sample; the upstream project's README
decision #2 admitted this once honestly, and this project must not repeat it.

## R5. `sharpe_max` is recorded only, never adjudicates

CLAUDE.md C5. The `sharpe_max` returned by the grid scan serves diagnosis
readability only. Any `if ... sharpe_max ...` selection logic is
parameter-level p-hacking.

## R6. You cannot see the test segment

The test segment is architecturally unreachable (`evaluate_series`'s `split`
accepts only train/val). Do not request it, infer it, or probe it indirectly
with phrasing like "after 2024". This is a design constraint, not a discipline.
