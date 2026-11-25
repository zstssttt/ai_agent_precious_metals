# Data Dictionary

The 15 factors in this repository consume 13 unique underlying indicators. Each is referenced by a stable `PMF_XXXX` code that lets the factor formulas depend on a code, not a file path, so vendors or file layouts can change without touching factor definitions.

## Indicator table

| PMF code | Indicator | Source | Native frequency | Purpose |
|----------|-----------|--------|:----------------:|---------|
| `PMF0001` | 10-year US Treasury nominal yield | FRED (`DGS10`) | Daily | Rates level |
| `PMF0002` | 10-year TIPS real yield | FRED (`DFII10`) | Daily | Real-yield proxy for opportunity cost of holding gold |
| `PMF0004` | COMEX gold futures close | Vendor | Daily | Global spot reference |
| `PMF0007` | Geopolitical Risk (GPR) index | [Caldara & Iacoviello](https://www.matteoiacoviello.com/gpr.htm) | Daily | Safe-haven demand proxy |
| `PMF0011` | Au9995 (Shanghai physical gold) close | Vendor | Daily | Strategy's traded asset |
| `PMF0012` | US Dollar Index (DXY) | Vendor / Yahoo (`DX-Y.NYB`) | Daily | Reserve-currency strength |
| `PMF0013` | US 10Y – 2Y Treasury spread | FRED (`T10Y2Y`) | Daily | Yield-curve shape / recession signal |
| `PMF0018` | Gold 1-month ATM option implied volatility | Vendor | Monthly | Forward-looking risk premium |
| `PMF0019` | Total gold ETF holdings (aggregated) | Vendor | Daily | Aggregate investor positioning |
| `PMF0020` | On/offshore gold futures basis | Vendor | Daily | Cross-market dislocation |
| `PMF0031` | SPDR Gold Shares monthly gold flow | Vendor | Monthly | Fast money creations/redemptions |
| `PMF0036` | SPDR Gold Shares monthly USD cash flow | Vendor | Monthly | Fast money creations/redemptions (dollar terms) |
| `PMF0042` | US core CPI year-over-year, seasonally adjusted | FRED (`CPILFESL`) | Monthly | Inflation regime |

## Frequency alignment

All weekly and monthly indicators are forward-filled onto the daily trading calendar upstream by [`src/data_preprocessing/align_to_trading_calendar.py`](../src/data_preprocessing/align_to_trading_calendar.py). Publication delays used:

| Native frequency | `DELAY` (calendar days) | Rationale |
|-----------------|:-----------------------:|-----------|
| Daily | 0 | Available end-of-day |
| Weekly | 3 | Publication lag for weekly commodity reports |
| Monthly | 7 | Publication lag typical for macro releases (e.g. CPI) |

After the shift, values are forward-filled onto every trading day, so factor code can treat every input as daily without knowing its native frequency.

## Factor formulas (headline)

Full formulas live in [`../references/factor_formulas.xlsx`](../references/factor_formulas.xlsx). One-line summaries:

### Dollar / Rates theme

- **`pmf_128453`** — rank-correlation change between DXY and SPDR cash flow, divided by GPR (risk-adjusted).
- **`pmf_011095`** — 5-day OLS slope of the rank-correlation series between DXY and SPDR flow return (identity).
- **`pmf_069545`** — direction from DXY 20d vs 60d MA, magnitude from |CPI 5d z-score|.

### Gold Price / Basis theme

- **`pmf_051113`** — direction from GPR 5d MA trend, magnitude from divergence of on/offshore basis 5d vs 20d mean.
- **`pmf_127111`** — Au9995 vs SPDR flow rank-correlation change, divided by option IV.
- **`pmf_125861`** — COMEX vs SPDR cash flow rank-correlation change, divided by GPR.

### ETF / Positioning theme

- **`pmf_131810`** — ETF holdings diff vs lagged 10Y yield return, 20-day correlation change, divided by GPR.
- **`pmf_104203`** — Aroon balance on SPDR flow returns × path persistence over 5 days.
- **`pmf_058341`** — SPDR flow 5d trend direction × (spread return / log|SPDR flow|).

### Inflation / Term structure theme

- **`pmf_102135`** — direction from CPI 5d vs 20d MA × CPI 1-lag autocorrelation persistence.
- **`pmf_003678`** — 20-day correlation change between TIPS diff and lagged ETF holdings diff (identity).
- **`pmf_118259`** — direction from CPI 20d vs 60d MA × CPI position within its 5-day Bollinger band.

### Risk / Volatility theme

- **`pmf_051056`** — GPR 5d trend direction × |CPI 20d return|.
- **`pmf_055491`** — option IV 20d trend direction × 5d mean of (|CPI diff| / log|SPDR cash flow|).
- **`pmf_066409`** — GPR 20d vs 60d MA direction × 5d mean of (|term spread| / log|SPDR flow|).
