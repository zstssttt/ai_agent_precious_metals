# Methodology

This document walks through every design decision behind the pipeline in the order they were made, so a reviewer can distinguish the choices that were derivable from ones that required judgment.

---

## 1. Problem framing

**Question:** *Given a universe of 50+ macro and market indicators, can we build a transparent, hypothesis-driven signal that predicts near-term direction of Au9995 (physical Chinese gold), and does that signal survive when we account for realistic trading frictions?*

Two constraints were fixed at the outset:

1. **No black boxes.** Every factor must have a written economic thesis and a formula a reviewer can read on one page. Deep learning and gradient boosting were explicitly out of scope. The mentor's rationale — one I have adopted for my own DS work since — is that a signal you cannot explain is a signal you cannot govern.
2. **Honest in-sample discipline.** Full-sample IC is used *only* to fix the trade direction; every other computation (rolling IC, ICIR, position sizing, portfolio weights) is either genuinely rolling or transparently disclosed as in-sample.

---

## 2. The five-theme framework

Instead of ranking factors by Sharpe and picking the top N, indicators are first bucketed into five economic themes. Each theme has a causal narrative about how it should move the gold price:

| Theme | Causal story | Representative indicators |
|-------|--------------|---------------------------|
| **Dollar / Rates** | Gold is USD-denominated. Higher real rates raise the opportunity cost of holding a non-yielding asset. | US Dollar Index (DXY), 10Y US Treasury yield |
| **Gold Price / Basis** | Cross-market dislocations (COMEX vs Au9995, futures vs spot) surface real-time supply/demand imbalances. | Au9995, COMEX gold close, on/offshore gold futures basis |
| **ETF / Positioning** | ETF creation/redemption is the highest-frequency observable measure of aggregate investor conviction. | Total gold ETF holdings, SPDR daily gold flow |
| **Inflation / Term structure** | Gold is a partial inflation hedge; the shape of the term structure hints at growth vs recession regimes. | 10Y TIPS real yield, US core CPI YoY, 10Y-2Y spread |
| **Risk / Volatility** | Safe-haven demand spikes when tail risk rises. | Geopolitical Risk index (GPR), gold 1-month ATM implied vol |

Three factors were designed per theme, giving 15 total. The framework serves two purposes:
- **Diversification by construction.** Combining themes with different economic exposures should reduce single-shock concentration.
- **Interpretability.** When a reviewer asks "why is the portfolio short today?", the answer is "the Dollar theme's `pmf_069545` flipped negative on the CPI z-score, and it currently carries 8 % of the risk budget."

---

## 3. Data engineering

### 3.1 Ingestion

Data comes from three sources: a Chinese commodity data vendor (Finoview) accessed via an authenticated internal API, FRED (Federal Reserve Economic Data), and Yahoo Finance for equity benchmarks. The crawler ([`src/data_ingest/finoview_crawler.py`](../src/data_ingest/finoview_crawler.py)) is idempotent by design:

- Any indicator already on disk is skipped, so a partial run resumes without duplicating downloads.
- Session credentials are read from environment variables — nothing sensitive is committed.
- Requests are rate-limited with a jittered 10–15 second sleep to be a polite API citizen.

### 3.2 The mixed-frequency problem

The 50+ raw indicators arrive at three different sampling frequencies: daily (yields, DXY, spot prices), weekly (some sentiment indices), and monthly (CPI, ETF flow totals). All downstream code expects a single unified daily time index against the Chinese trading calendar.

Naive resampling would introduce two subtle bugs:

- **Look-ahead in weekly/monthly indicators.** A monthly CPI value for June is not knowable until early July when it is released. Forward-filling from the reference date would let a factor "see" a value on days when the market did not have it.
- **Trading-calendar misalignment.** Vendor CSVs use calendar days; the trading calendar has weekends and public holidays removed.

The aligner ([`src/data_preprocessing/align_to_trading_calendar.py`](../src/data_preprocessing/align_to_trading_calendar.py)) handles both:

1. Every reference date is shifted forward by a `DELAY` parameter (0 for daily, 3 for weekly, 7 for monthly — matching typical publication lags).
2. The shifted series is forward-filled onto a contiguous daily grid.
3. Non-trading days are dropped so the output only contains dates where a strategy could actually trade.

A per-file audit is emitted (`processing_log.csv`) tracking `added_rows / deleted_rows / nan_count / final_rows`, so any silent data loss surfaces immediately.

### 3.3 The date-column bug I diagnosed but did not merge

The 2.0 alignment script was silently producing empty CSVs. Root cause: the date-column detector iterated columns in appearance order and matched anything containing "date" or "time". Some vendor files carried a `time` column (all zeros) *before* the true `date` column; `pd.to_datetime(0)` returns 1970-01-01, and the subsequent trading-calendar filter dropped every row. Documented in [`docs/methodology_debug_notes.md`](methodology_debug_notes.md); the fix (prioritize `date` over `time`, use an intermediate column name to avoid overwriting the original) was left for the team to review before merging.

---

## 4. Exploratory QC — before any factor is written

Two scripts run over the aligned indicator lake before a single factor line is written:

- [`src/exploratory_analysis/distribution_qc.py`](../src/exploratory_analysis/distribution_qc.py) — per-indicator histogram, Q-Q plot (against a normal), and time series. If an indicator has a bimodal distribution or a suspicious regime break, we catch it here.
- [`src/exploratory_analysis/correlation_analysis.py`](../src/exploratory_analysis/correlation_analysis.py) — pairwise Pearson correlation, hierarchically clustered so that redundant indicators (e.g. multiple SPDR flow variants) visibly cluster together on the heatmap. Any pair with |ρ| > 0.7 is written into a CSV so factor design can prune duplicates.

This step is what separates a report that reads *"we built 15 factors"* from *"here is a data lake I actually looked at"*.

---

## 5. Factor library

Every factor is a small pure function of the form `dict[str, pd.Series] -> pd.Series` living in [`src/factors/build_factors.py`](../src/factors/build_factors.py). Under the hood it composes two layers of primitives:

### 5.1 Rolling time-series operators (bricks)

| Operator | Definition | Financial use |
|----------|------------|---------------|
| `ts_delay(x, n)` | Lag by n periods | Historical values, day-over-day changes |
| `ts_mean(x, n)` | Rolling mean | Trend / moving average |
| `ts_std(x, n)` | Rolling std | Volatility, z-score denominator |
| `ts_sum(x, n)` | Rolling sum | Cumulative move |
| `ts_rank(x, n)` | Rolling rank | Removes scale differences; enables Spearman-style analysis |
| `ts_corr_binary(x, y, n)` | Rolling Pearson ρ | Co-movement between two series |
| `ts_linreg_slope(x, n)` | OLS slope of x on t=[0..n-1] | Robust trend estimate |
| `ts_aroon_balance(x, n)` | Aroon Up − Aroon Down | Recency of highs vs lows (momentum flavor) |

All rolling functions use `min_periods = n` — no fractional-window approximations. `zero2nan` is used at every division site so that pathological zero denominators produce NaN rather than infinity.

### 5.2 Composition templates (paradigms)

Every one of the 15 factors reduces to one of five paradigms:

| Template | Formula shape | Economic reading |
|----------|---------------|-------------------|
| `identity` | *expr* | Take the expression as-is |
| `risk_adjusted` | *signal / risk* | Discount signals during high-risk regimes |
| `direction × intensity` | sign(*a*) × \|*b*\| | Direction from one series, magnitude from another |
| `signal × persistence` | *signal* × persistence(*signal*) | Reward signals that don't flip-flop |
| `direction × position` | sign(*a*) × position(*b*) | Direction × where the second series sits in its band |

Naming the templates explicitly was one of the highest-leverage refactors of the project: it turned a wall of formulas into 5 patterns × 3 examples, which is what a reviewer can hold in their head.

---

## 6. IC diagnostics

### 6.1 What is IC?

*Information Coefficient* = rank correlation between today's factor value and the forward n-day return. Its running mean (IC) tells you whether the factor systematically points in the right direction; its running standard deviation gives you a t-statistic (ICIR = IC / std(IC)) that quantifies how noisy that pointing is.

### 6.2 Choices made here

- **Rolling 180-day window.** Full-sample IC hides regime changes; a shorter window (60-90d) is too noisy to reject the null. 180 days is the horizon over which factor–return relationships tend to remain stationary in metals markets.
- **Spearman rank IC as the default.** Robust to outliers and non-linear monotonic relationships; a Pearson IC is also computed and heatmapped side-by-side.
- **10 forward horizons Y₁ … Y₁₀.** Different factors work at different horizons; the heatmap ([`results/plots/rolling_IC_heatmap.png`](../results/plots/rolling_IC_heatmap.png)) makes this immediately visible.
- **Significance stars overlaid.** ICIR ≥ 0.5 gets `*`, ≥ 1.0 gets `**`, so a reader can distinguish "big IC on 6 observations" from "small IC on 8,000 observations".

---

## 7. Backtest and portfolio construction

### 7.1 Position mapping — three styles

Each factor's raw value is mapped to a target position in [-1, +1] using **exactly one** of three styles, all rolling over a 252-day window:

- **`rolling_percentage`** — where does today's value rank within the last 252 days? Percentile → linear map to [-1, +1]. Robust to outliers; extreme values collapse to ±1 without ad-hoc clipping.
- **`rolling_zscore_tanh`** — z-score of today's value, squashed through tanh. Smoothly bounded.
- **`rolling_zscore_clip`** — z-score, hard-clipped to [-1, +1]. Harder-edged than tanh.

Percentile came out on top in every backtest; the three are reported side-by-side to make the choice defensible instead of hand-tuned.

### 7.2 Post-processing — the TWAP overlay

After mapping, every position series passes through:

```
pos = apply_holding_period(pos, HOLDING_PERIOD)   # step 1: sparsify to every 10 days
pos = smooth_position(pos, POSITION_SMOOTH_WINDOW) # step 2: 5-day rolling mean
```

Step 1 turns the daily signal into a step function that only changes every 10 days. Step 2 linearly interpolates each step over 5 days, producing a TWAP-style execution overlay. In the linear-cost model used here, total turnover is preserved but spread out.

**Testing showed reversing the two lines drops portfolio Sharpe by ~0.2.** The step-then-smooth order is essential: it means the smoother is executing rebalances gradually, not blurring the signal itself.

### 7.3 Portfolio construction — Sharpe-weighted, not IC-weighted

Three weighting schemes were tested:

| Scheme | Sharpe (best mapping) | Comment |
|--------|----------------------:|---------|
| Equal-weight all 15 | 0.40 | 6 of 15 factors had \|IC\| < 0.01, diluting the signal |
| IC-weighted | 0.26 | The highest-\|IC\| factor produced NaN positions (all-CPI factors on monthly data); allocating to it gave weight without exposure |
| **Sharpe-weighted (floor=0)** | **0.94** | Uses the only metric that has actually been validated |

Sharpe-weighted uses `weight_i = max(single_sharpe_i - SHARPE_FLOOR, 0) / sum(...)`. `SHARPE_FLOOR = 0.0` (the default) drops any factor with negative in-sample Sharpe; raising it to 0.3 gives fewer, higher-quality factors but *lower* portfolio Sharpe because the excluded factors (single-name Sharpe 0.13-0.14) still contributed diversification worth more than the noise they added.

### 7.4 Execution and cost model

- **T+1 execution.** Position at close of day `t-1` earns the return between close `t-1` and close `t`, so no signal ever sees its own realized return.
- **Compounded NAV.** `nav = (1 + net).cumprod()`, not `1 + net.cumsum()`. The additive form went negative under a >100 % drawdown in the earlier version, producing the infamous "162 % max drawdown" that we spent an evening debugging.
- **Linear cost of 3 bp per unit of turnover.** Realistic for the gold market; not aggressive.

### 7.5 The double-smoothing bug that cost us 0.6 in Sharpe

The first Sharpe-weighted implementation applied `postprocess_position` to each single-factor position, then again to the weighted sum. Two rounds of a 5-day rolling mean over-smoothed the portfolio signal into near-flat exposure and dropped Sharpe from what should have been 0.94 down to 0.35. Fix: cache the per-factor positions *before* post-processing, weight-sum, then post-process once.

**Takeaway:** any pipeline that composes stages should be able to answer "does this stage assume the input is raw or already processed?" for every stage. Naming the intermediate objects (`directional_positions` vs `final_positions`) makes the answer physical.

---

## 8. Benchmarking — why vol-normalized CSI 300

The strategy runs at 5.5 % annualized volatility. Comparing it to Au9995 buy-and-hold (16 %) or raw CSI 300 (26 %) is silently comparing to a 3–5 × leveraged reference; absolute-return comparisons make the strategy look weak for reasons that have nothing to do with signal quality.

The fix: scale the benchmark's *daily returns* to a common volatility target (`r' = r × target_vol / raw_vol`), then compound. Scaling an already-built NAV curve geometrically preserves shape but does *not* preserve annualized volatility — this trap swallowed an hour of debugging before we caught it.

At the same 5.5 % vol budget, the strategy compounds ~3.8 × the CSI 300 benchmark and delivers ~2.4 × its Sharpe.

---

## 9. Honest limitations

- **In-sample direction.** Full-sample IC is used to fix trade direction. Genuinely rolling-IC direction was tested and drops Sharpe to ~0.30, giving a lower bound on what a truly out-of-sample deployment would look like.
- **Single asset.** The signal is defined only on Au9995; cross-asset generalization is not tested.
- **6 of 15 factors did most of the work.** The three CPI-driven factors (`pmf_069545`, `pmf_102135`, `pmf_118259`) produce NaN positions because forward-filled monthly CPI has zero rolling standard deviation over 5-day windows. This is a genuine data-frequency limitation and would require a different position-mapping strategy for those factors.
- **In-sample Sharpe ≠ live Sharpe.** The 0.94 figure is a diagnostic on 2015-2023 data. A truly out-of-sample forward-Sharpe on gold factors of this style typically lands in 0.3–0.5.

---

## 10. What I would do differently next time

1. **Move indicator selection into the data lake.** The 50+ indicators were curated by hand into 13 for the top-15 factors. A more scalable approach would be to auto-select from the correlation-cluster output, which would make the process more reproducible and less mentor-anchored.
2. **Walk-forward IC direction.** Even if it lowers reported Sharpe, using a rolling window for the direction decision would remove the last in-sample assumption.
3. **Non-linear cost model.** Once the pipeline moves to genuinely tradeable capital, a market-impact-aware cost model (`cost ~ turnover^1.5`) would make the TWAP smoother pay for itself.
4. **Automated factor factory.** The 5 composition templates + 8 primitives combinatorially produce hundreds of candidate factors. A future version would sweep them with proper multiple-hypothesis correction, not hand-select 15.
