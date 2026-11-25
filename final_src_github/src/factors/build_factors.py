"""Main engine script for computing the Top-15 gold price factors.

Reads 13 PMF-coded macro/market indicators from the preprocessed indicator
directory (these indicators have already been forward-filled to the daily
Chinese trading calendar by ``process_delayed_data.py``), evaluates 15 factor
formulas across 5 economic themes (Dollar / Gold Price / ETF / Rates / Risk),
and outputs a wide-format CSV whose column names are the factor codes, ready
for direct consumption by downstream ``factor_ic_analysis.py`` and
``position_mapping.py``.

Usage:
    python factor_top15/3.0factor_top15_en.py             # Full run, output goes to OUTPUT_PATH
    python factor_top15.py --dry-run          # Only verify that PMF file mappings are correct
    python factor_top15.py --factors pmf_X    # Compute only the specified factor (smoke test)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Configuration section (adjust these paths before running)
# ======================================================================

import os

# Root directory containing the preprocessed indicator files. Override with the
# GOLD_FACTOR_ALIGNED_ROOT env var when running against a different data mount.
INDICATOR_ROOT = Path(
    os.environ.get("GOLD_FACTOR_ALIGNED_ROOT", "data/aligned")
)

# Output path for the factor computation results.
OUTPUT_PATH = Path(
    os.environ.get(
        "FACTOR_OUTPUT_PATH",
        "results/factor_values_top15.csv",
    )
)

# Data cutoff date — keep only data on or before this date
CUTOFF_DATE = "2023-12-31"

# PMF code → file path relative to INDICATOR_ROOT.
# Only the 13 indicators actually used by the 15 factor formulas are listed here.
PMF_TO_FILENAME: dict[str, str] = {
    "PMF0001": "补齐后日频数据2.0/10年美国国债收益率.csv",                       # 10-year US Treasury yield
    "PMF0002": "补齐后日频数据2.0/10年美国国债收益率通胀指数国债(TIPS).csv",       # 10-year TIPS real yield
    "PMF0004": "补齐后日频数据2.0/COMEX黄金_收盘价_S0180903.csv",               # COMEX gold closing price
    "PMF0007": "补齐后日频数据2.0/提取后地缘政治.csv",                           # Geopolitical Risk Index (GPR)
    "PMF0011": "补齐后日频数据2.0/收盘价_Au9995_S0035818.csv",             # Au9995 gold spot closing price
    "PMF0012": "补齐后日频数据2.0/美元指数_M0000271.csv",                       # US Dollar Index (DXY)
    "PMF0013": "补齐后日频数据2.0/美国10年期和2年期国债利差_USYC2Y10 INDEX.csv",  # 10Y-2Y term spread
    "PMF0018": "补齐后月频数据/黄金1个月平价期权波动率.csv",                      # Gold 1-month at-the-money option implied volatility
    "PMF0019": "补齐后日频数据2.0/黄金ETF总持有量.csv",                          # Total gold ETF holdings
    "PMF0020": "补齐后日频数据2.0/黄金_金融市场_内外期货价差.csv",                # Domestic-vs-overseas gold futures spread
    "PMF0031": "补齐后日频数据2.0/SPDR/SPDR黄金ETF持有量.csv",    # Changed to SPDR gold ETF holdings.csv
    "PMF0036": "补齐后日频数据2.0/SPDR/SPDR黄金ETF持有量.csv", # Changed to SPDR gold ETF holdings.csv
    "PMF0042": "补齐后月频数据/美国核心CPI_季调_当月同比_G0000030 2.xlsx",        # US core CPI YoY (seasonally adjusted)
}


# ======================================================================
# Constants
# ======================================================================

EPSILON = 1e-8  # Lower bound for denominators to guard against division by zero


# ======================================================================
# Part 1: Basic time-series operators
# ======================================================================
# These are the "building block" functions used by all factor formulas. Each
# takes a pd.Series and window parameter(s) and returns a pd.Series of the
# same length. For rolling-window functions, min_periods equals the window
# length, meaning NaN is emitted when the window is not fully populated —
# no approximation on partial windows.

def ts_delay(x: pd.Series, n: int) -> pd.Series:
    """Time-series delay: shift the series backward by n steps (value at time t is the value at t-n).

    Example: ts_delay([10, 20, 30, 40, 50], 2) → [NaN, NaN, 10, 20, 30]
    Financial meaning: retrieve historical values, commonly used to compute changes (today - yesterday).
    """
    return x.shift(n)


def ts_mean(x: pd.Series, n: int) -> pd.Series:
    """Rolling mean: arithmetic average over the past n periods.

    Example: ts_mean([10, 20, 30, 40, 50], 3) → [NaN, NaN, 20, 30, 40]
    Financial meaning: moving average, used to identify trend direction and denoise.
    """
    return x.rolling(n, min_periods=n).mean()


def ts_std(x: pd.Series, n: int) -> pd.Series:
    """Rolling standard deviation: sample standard deviation over the past n periods.

    Example: ts_std([10, 20, 30, 40, 50], 3) → [NaN, NaN, 10, 10, 10]
    Financial meaning: measures recent volatility, used for Bollinger bands and z-score calculations.
    """
    return x.rolling(n, min_periods=n).std()


def ts_sum(x: pd.Series, n: int) -> pd.Series:
    """Rolling sum: cumulative sum over the past n periods.

    Example: ts_sum([1, 2, 3, 4, 5], 3) → [NaN, NaN, 6, 9, 12]
    Financial meaning: quantifies cumulative movement over a time span.
    """
    return x.rolling(n, min_periods=n).sum()


def ts_rank(x: pd.Series, n: int) -> pd.Series:
    """Rolling rank: the current value's rank within the past n-period window.

    Example: window [3, 1, 4, 1, 5], current value 5 → rank 5 (largest)
    Financial meaning: converts raw values into "relative position within recent
    history", removing scale differences. Computing corr on ranked series is
    essentially Spearman rank correlation.
    """
    return x.rolling(n, min_periods=n).rank()


def ts_corr_binary(x: pd.Series, y: pd.Series, n: int) -> pd.Series:
    """Rolling Pearson correlation: linear correlation between two series over the past n periods.

    Range [-1, 1]; 1 means perfect positive correlation, -1 means perfect negative correlation.
    Financial meaning: quantifies the recent co-movement between two assets/indicators.
    """
    return x.rolling(n, min_periods=n).corr(y)


def ts_linreg_slope(x: pd.Series, n: int) -> pd.Series:
    """Rolling linear regression slope: OLS regression over the past n periods with independent variable [0, 1, ..., n-1].

    Slope > 0 indicates the series is rising within the window; < 0 indicates falling.
    More robust than a simple "end - start" difference because it uses all intermediate points.

    Implementation details:
    - t = [0, 1, 2, ..., n-1]; compute the mean and variance of t (fixed, computed once)
    - For each window, compute the mean of x and the covariance between x and t
    - slope = cov(t, x) / var(t)
    """
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum() / n  # Population variance (divided by n, not n-1)

    def _slope(window: np.ndarray) -> float:
        x_mean = window.mean()
        cov = ((t - t_mean) * (window - x_mean)).sum() / n
        return cov / t_var if t_var != 0 else np.nan

    return x.rolling(n, min_periods=n).apply(_slope, raw=True)


def ts_aroon_balance(x: pd.Series, n: int) -> pd.Series:
    """Aroon long/short balance: over the past n+1 periods, how recent the high is vs how recent the low is.

    Aroon Up   = (n - periods since the high) / n, range [0, 1]
    Aroon Down = (n - periods since the low)  / n, range [0, 1]
    Balance    = Aroon Up - Aroon Down, range [-1, 1]

    Balance > 0 → a recent new high (bullish)
    Balance < 0 → a recent new low (bearish)

    Example: window = [10, 15, 12, 8, 14], n = 4
    Highest at index=1 (value 15), 3 days ago → Aroon Up = (4-3)/4 = 0.25
    Lowest at index=3 (value 8), 1 day ago → Aroon Down = (4-1)/4 = 0.75
    Balance = 0.25 - 0.75 = -0.5 (bears dominate because the low occurred more recently)
    """
    def _balance(window: np.ndarray) -> float:
        last_high_idx = int(np.argmax(window))     # Position of the first occurrence of the max in the window
        last_low_idx = int(np.argmin(window))      # Position of the first occurrence of the min in the window
        periods_since_high = (len(window) - 1) - last_high_idx  # Days since the high
        periods_since_low = (len(window) - 1) - last_low_idx    # Days since the low
        aroon_up = (n - periods_since_high) / n
        aroon_down = (n - periods_since_low) / n
        return aroon_up - aroon_down

    return x.rolling(n + 1, min_periods=n + 1).apply(_balance, raw=True)


def zero2nan(x):
    """Replace zeros with NaN. Accepts a scalar, Series, or DataFrame.

    Purpose: guard division operations. When the denominator might be 0,
    apply zero2nan first and then divide; the result is NaN (missing) rather
    than inf (infinity), which is safer for downstream processing.
    """
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.where(x != 0, np.nan)
    return np.nan if x == 0 else x


# ======================================================================
# Part 2: Composition templates (factor combination paradigms)
# ======================================================================
# Although the 15 factors differ, all fall into one of the following 5
# "composition paradigms". The basic operators are the bricks; these
# templates are the assembly instructions.

def apply_identity(expression: pd.Series) -> pd.Series:
    """Identity template: the expression itself is the factor value, no transformation applied."""
    return expression


def apply_risk_adjusted(numerator: pd.Series, risk: pd.Series) -> pd.Series:
    """Risk-adjusted template: raw signal ÷ risk denominator (NaN-protected when denominator is zero).

    Economic meaning: the same signal strength should be "discounted" in a
    high-risk environment because everything moves during high-risk periods
    and the signal-to-noise ratio is lower.
    """
    return numerator / zero2nan(risk)


def apply_direction_x_intensity(direction: pd.Series, intensity: pd.Series) -> pd.Series:
    """Direction × intensity template: use one signal to determine direction (sign), and another to quantify intensity.

    Economic meaning: first answer "up or down?" (direction), then answer "how strong?" (intensity).
    The direction signal after sign() takes only +1/-1/NaN; the intensity uses the absolute value to ensure non-negativity.
    """
    return zero2nan(np.sign(direction)) * np.abs(intensity)


def apply_signal_x_persistence(signal: pd.Series, persistence: pd.Series) -> pd.Series:
    """Signal × persistence template: the signal value multiplied by a measure of the signal's own persistence.

    Economic meaning: a signal that keeps flip-flopping is unreliable;
    signals with high persistence deserve greater weight.
    """
    return signal * persistence


def apply_direction_x_position(direction: pd.Series, position: pd.Series) -> pd.Series:
    """Direction × position template: the sign of the direction signal, multiplied by a 0-1 position indicator.

    Economic meaning: first determine direction, then look at where the
    indicator sits within its own volatility range (e.g., high or low within
    a Bollinger band).
    """
    return zero2nan(np.sign(direction)) * position


# ======================================================================
# Part 3: Data loading utilities
# ======================================================================

def read_csv_any_encoding(path: str | Path, **kwargs) -> pd.DataFrame:
    """Try reading a CSV in the order utf-8 → gbk → gb2312 → latin1.

    Encoding of Chinese CSV files is inconsistent; different sources use
    different encodings. This function tries four common encodings in order
    and uses whichever succeeds.
    """
    for encoding in ("utf-8", "gbk", "gb2312", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Unable to decode file with any known encoding: {path}")


def _parse_date_price_df(
    df: pd.DataFrame, path: str, rename_price_to: str | None = None
) -> pd.DataFrame:
    """Standardize a raw DataFrame into a date-indexed, single-price-column format.

    Processing steps:
    1. Strip whitespace from column names
    2. Case-insensitively find the "date" and "price" columns
    3. Convert the date column to datetime and set as the index
    4. Sort by date and drop duplicates (keep the last record when a date has multiple)
    5. Rename the price column to the PMF code
    """
    df.columns = df.columns.str.strip()
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    price_col = next((c for c in df.columns if c.lower() == "price"), None)
    if date_col is None:
        raise ValueError(f"'date' column not found in file: {path}")
    if price_col is None:
        raise ValueError(f"'price' column not found in file: {path}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()[[price_col]]
    df = df[~df.index.duplicated(keep="last")]  # Deduplicate: keep the last occurrence
    if rename_price_to:
        df = df.rename(columns={price_col: rename_price_to})
    return df


def load_indicator(
    pmf_code: str, indicator_root: str | Path, pmf_map: dict[str, str]
) -> pd.Series:
    """Load a single indicator by PMF code and return a Series named by that code.

    Steps:
    1. Look up the relative file path from the mapping table
    2. Build the full path; choose read_csv or read_excel based on the file extension
    3. Standardize into a date-indexed numeric Series (e.g., PMF0001)
    """
    if pmf_code not in pmf_map:
        raise KeyError(f"No file mapping found for {pmf_code}")
    filepath = Path(indicator_root) / pmf_map[pmf_code]
    if not filepath.exists():
        raise FileNotFoundError(f"File for {pmf_code} does not exist: {filepath}")
    if filepath.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        df = read_csv_any_encoding(filepath)
    df = _parse_date_price_df(df, str(filepath), rename_price_to=pmf_code)
    return pd.to_numeric(df[pmf_code], errors="coerce")


def load_all_indicators(
    indicator_root: str | Path, pmf_map: dict[str, str]
) -> dict[str, pd.Series]:
    """Load all mapped indicators and return a {PMF code: Series} dictionary.

    On file-missing or parse errors, skip that indicator and log a warning
    without interrupting the overall flow.
    """
    series_by_code: dict[str, pd.Series] = {}
    for pmf_code in pmf_map:
        try:
            series_by_code[pmf_code] = load_indicator(pmf_code, indicator_root, pmf_map)
            logger.info(
                "Loaded %s (%d valid observations, %s → %s)",
                pmf_code,
                series_by_code[pmf_code].notna().sum(),
                series_by_code[pmf_code].index.min().date(),
                series_by_code[pmf_code].index.max().date(),
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.warning("Skipping %s: %s", pmf_code, exc)
    return series_by_code


# ======================================================================
# Part 4: Top-15 factor definitions
# ======================================================================
# Grouped by economic theme (5 themes × 3 factors each); within each theme,
# factors are sorted from highest to lowest Sharpe ratio. Each function takes
# a {PMF code: Series} dictionary and returns a Series named by the factor
# code.
#
# Formulas are transcribed verbatim from top15_因子公式.xlsx. Repeated
# sub-expressions are bound once to local variables; numeric constants are
# kept as their original values without abstraction.


# ---- Theme 1: Dollar ----

def factor_dollar_1_pmf_128453(inds: dict[str, pd.Series]) -> pd.Series:
    """Correlation change between DXY and SPDR gold cash flow, divided by geopolitical risk for risk adjustment.

    Composition template: risk_adjusted
    Indicators used: PMF0012 (DXY US Dollar Index), PMF0036 (SPDR monthly cash flow), PMF0007 (Geopolitical Risk / GPR)

    Logic:
    1. Compute daily changes of DXY and SPDR cash flow and rank each within a 5-day window (rank transform)
    2. Compute the 5-day rolling Pearson correlation on the ranked series (essentially Spearman rank correlation)
    3. Take the 5-day change of the correlation as the signal (is co-movement strengthening or weakening?)
    4. Divide by GPR for risk normalization
    """
    dxy_diff_rank = ts_rank(inds["PMF0012"] - ts_delay(inds["PMF0012"], 1), 5)
    spdr_cash_diff_rank = ts_rank(inds["PMF0036"] - ts_delay(inds["PMF0036"], 1), 5)
    corr_5d = ts_corr_binary(dxy_diff_rank, spdr_cash_diff_rank, 5)
    numerator = corr_5d - ts_delay(corr_5d, 5)
    result = 1 * numerator / zero2nan(inds["PMF0007"])
    result.name = "pmf_128453"
    return result


def factor_dollar_2_pmf_011095(inds: dict[str, pd.Series]) -> pd.Series:
    """5-day OLS slope of the correlation series between DXY and SPDR gold flow returns.

    Composition template: identity
    Indicators used: PMF0012 (DXY), PMF0031 (SPDR gold monthly flow)

    Logic:
    1. Take daily change of DXY and rank over 5 days
    2. Take daily return of SPDR flow (today/yesterday - 1) and rank over 5 days
    3. Compute the 5-day rolling correlation between the two rank series
    4. Fit a 5-day linear regression on the correlation series and take the slope
       slope > 0 → the co-movement between DXY and SPDR is trending upward
    """
    dxy_diff_rank = ts_rank(inds["PMF0012"] - ts_delay(inds["PMF0012"], 1), 5)
    spdr_ret_rank = ts_rank(
        inds["PMF0031"] / zero2nan(ts_delay(inds["PMF0031"], 1)) - 1, 5
    )
    corr_5d = ts_corr_binary(dxy_diff_rank, spdr_ret_rank, 5)
    result = 1 * ts_linreg_slope(corr_5d, 5)
    result.name = "pmf_011095"
    return result


def factor_dollar_3_pmf_069545(inds: dict[str, pd.Series]) -> pd.Series:
    """DXY 20-day MA vs 60-day MA determines direction, multiplied by |5-day CPI z-score| as intensity.

    Composition template: direction_x_intensity
    Indicators used: PMF0012 (DXY), PMF0042 (US core CPI YoY)

    Logic:
    1. Sign of (20-day MA of DXY daily diff) minus (60-day MA of DXY daily diff) → direction (+1 or -1)
       MA20 > MA60 → short-term above long-term → dollar strengthening trend
    2. Absolute value of the 5-day z-score of CPI → intensity
       z-score = (current value - 5-day mean) / 5-day std
       Larger absolute value → CPI deviates more from its recent mean → stronger inflation signal
    3. direction × intensity = factor value
    """
    dxy_diff = inds["PMF0012"] - ts_delay(inds["PMF0012"], 1)
    direction = zero2nan(np.sign(ts_mean(dxy_diff, 20) - ts_mean(dxy_diff, 60)))
    cpi_zscore = (inds["PMF0042"] - ts_mean(inds["PMF0042"], 5)) / zero2nan(
        ts_std(inds["PMF0042"], 5)
    )
    result = 1 * direction * np.abs(cpi_zscore)
    result.name = "pmf_069545"
    return result


# ---- Theme 2: Gold Price / Relative Spread ----

def factor_goldprice_1_pmf_051113(inds: dict[str, pd.Series]) -> pd.Series:
    """GPR 5-day trend direction, multiplied by the mean-divergence intensity of the domestic-vs-overseas gold spread return.

    Composition template: direction_x_intensity
    Indicators used: PMF0007 (GPR), PMF0020 (domestic-vs-overseas gold futures spread)

    Logic:
    1. Compare the 5-day MA of GPR with its value 5 days ago; take the sign → direction
       GPR rising → geopolitical risk intensifying
    2. Absolute value of (5-day MA - 20-day MA) of the spread's daily return → intensity
       The more the 5-day mean deviates from the 20-day mean → the more short-term spread behavior diverges from mid-term → pricing anomaly
    """
    gpr_ma5 = ts_mean(inds["PMF0007"], 5)
    direction = zero2nan(np.sign(gpr_ma5 - ts_delay(gpr_ma5, 5)))
    spread_ret = inds["PMF0020"] / zero2nan(ts_delay(inds["PMF0020"], 1)) - 1
    intensity = np.abs(ts_mean(spread_ret, 5) - ts_mean(spread_ret, 20))
    result = 1 * direction * intensity
    result.name = "pmf_051113"
    return result


def factor_goldprice_2_pmf_127111(inds: dict[str, pd.Series]) -> pd.Series:
    """Correlation change between Au9995 and SPDR flow, divided by option implied volatility for risk adjustment.

    Composition template: risk_adjusted
    Indicators used: PMF0011 (Au9995 close), PMF0031 (SPDR flow), PMF0018 (option implied volatility)

    Logic:
    1. Take daily change of Au9995 and SPDR flow and rank each over 5 days
    2. Compute the 5-day rolling correlation on the rank series
    3. Take the 5-day change of the correlation → signal
    4. Divide by option implied volatility → the signal is "shrunk" during high volatility
       When volatility is high, everything is moving in the market and correlation change may not be meaningful
    """
    au_diff_rank = ts_rank(inds["PMF0011"] - ts_delay(inds["PMF0011"], 1), 5)
    spdr_diff_rank = ts_rank(inds["PMF0031"] - ts_delay(inds["PMF0031"], 1), 5)
    corr_5d = ts_corr_binary(au_diff_rank, spdr_diff_rank, 5)
    numerator = corr_5d - ts_delay(corr_5d, 5)
    result = 1 * numerator / zero2nan(inds["PMF0018"])
    result.name = "pmf_127111"
    return result


def factor_goldprice_3_pmf_125861(inds: dict[str, pd.Series]) -> pd.Series:
    """Correlation change between COMEX gold and SPDR cash flow, divided by geopolitical risk for risk adjustment.

    Composition template: risk_adjusted
    Indicators used: PMF0004 (COMEX gold close), PMF0036 (SPDR cash flow), PMF0007 (GPR)

    Logic: same structure as pmf_128453, but the price-side indicator is swapped:
    - pmf_128453 uses DXY vs SPDR cash flow
    - here we use COMEX gold price vs SPDR cash flow
    Both examine how "price vs fund-flow co-movement changes" behave under different risk environments.
    """
    comex_diff_rank = ts_rank(inds["PMF0004"] - ts_delay(inds["PMF0004"], 1), 5)
    spdr_cash_diff_rank = ts_rank(inds["PMF0036"] - ts_delay(inds["PMF0036"], 1), 5)
    corr_5d = ts_corr_binary(comex_diff_rank, spdr_cash_diff_rank, 5)
    numerator = corr_5d - ts_delay(corr_5d, 5)
    result = 1 * numerator / zero2nan(inds["PMF0007"])
    result.name = "pmf_125861"
    return result


# ---- Theme 3: ETF / Holdings / Liquidity ----

def factor_etf_1_pmf_131810(inds: dict[str, pd.Series]) -> pd.Series:
    """20-day correlation change between ETF holdings change and 10Y Treasury yield (lag 1), divided by GPR for risk adjustment.

    Composition template: risk_adjusted
    Indicators used: PMF0019 (total gold ETF holdings), PMF0001 (10Y US Treasury yield), PMF0007 (GPR)

    Logic:
    1. Daily change of ETF holdings
    2. Daily return of the 10Y Treasury yield, lagged by 1 period
       (Lag because the bond market may lead ETF reaction by 1 day)
    3. Compute a 20-day rolling correlation, then take the 20-day change → signal
    4. Divide by GPR → risk adjustment

    Note this factor uses a longer window (20 days) to capture medium-term
    co-movement change, whereas the earlier dollar/gold-price factors use a
    5-day window and are more short-term.
    """
    etf_diff = inds["PMF0019"] - ts_delay(inds["PMF0019"], 1)
    yield_ret_lag = ts_delay(
        inds["PMF0001"] / zero2nan(ts_delay(inds["PMF0001"], 1)) - 1, 1
    )
    corr_20d = ts_corr_binary(etf_diff, yield_ret_lag, 20)
    numerator = corr_20d - ts_delay(corr_20d, 20)
    result = 1 * numerator / zero2nan(inds["PMF0007"])
    result.name = "pmf_131810"
    return result


def factor_etf_2_pmf_104203(inds: dict[str, pd.Series]) -> pd.Series:
    """Aroon long/short balance × path-persistence ratio on SPDR flow returns.

    Composition template: signal_x_persistence
    Indicators used: PMF0031 (SPDR gold monthly flow)

    Logic:
    1. Compute the daily return of SPDR flow
    2. Signal = Aroon balance (5-day): whether recent returns are hitting new highs (bullish) or new lows (bearish)
    3. Persistence = |5-day total change| / sum of |daily changes| over 5 days
       - If moving in the same direction every day → numerator ≈ denominator → persistence ≈ 1
       - If oscillating back and forth → numerator << denominator → persistence << 1
       This is like asking "did you walk this path straight, or zigzag through it?"
    4. signal × persistence = factor value
       Aroon says direction; persistence says "is that direction reliable?"
    """
    spdr_ret = inds["PMF0031"] / zero2nan(ts_delay(inds["PMF0031"], 1)) - 1
    signal = ts_aroon_balance(spdr_ret, 5)
    numerator = np.abs(spdr_ret - ts_delay(spdr_ret, 5))      # Absolute value of the 5-day total change
    denominator = ts_sum(np.abs(spdr_ret - ts_delay(spdr_ret, 1)), 5)  # Sum of absolute daily changes
    persistence = numerator / zero2nan(denominator)
    result = 1 * signal * persistence
    result.name = "pmf_104203"
    return result


def factor_etf_3_pmf_058341(inds: dict[str, pd.Series]) -> pd.Series:
    """SPDR flow 5-day trend direction × intensity of term-spread return / log|SPDR|.

    Composition template: direction_x_intensity
    Indicators used: PMF0031 (SPDR flow), PMF0013 (US 10Y-2Y Treasury term spread)

    Logic:
    1. 5-day MA of SPDR flow vs its value 5 days ago; take the sign → direction (are funds flowing in or out?)
    2. Intensity = 5-day mean of (US 10Y-2Y term spread return) / log|SPDR flow|
       - Spread return measures bond-market volatility
       - Dividing by log|SPDR| normalizes by the ETF's size
       - Take a 5-day mean to smooth
    """
    spdr_ma5 = ts_mean(inds["PMF0031"], 5)
    direction = zero2nan(np.sign(spdr_ma5 - ts_delay(spdr_ma5, 5)))
    spread_abs_ret = np.abs(
        inds["PMF0013"] / zero2nan(ts_delay(inds["PMF0013"], 1)) - 1
    )
    log_spdr = np.abs(np.log(zero2nan(np.abs(inds["PMF0031"]))))
    intensity = ts_mean(spread_abs_ret / zero2nan(log_spdr), 5)
    result = 1 * direction * intensity
    result.name = "pmf_058341"
    return result


# ---- Theme 4: Rates / Inflation / Term Structure ----

def factor_rates_1_pmf_102135(inds: dict[str, pd.Series]) -> pd.Series:
    """Direction of CPI 5-day MA vs 20-day MA × 1st-order autocorrelation persistence of CPI itself.

    Composition template: signal_x_persistence
    Indicators used: PMF0042 (US core CPI YoY)

    Logic:
    1. Signal = sign(MA5 - MA20): CPI short-term MA above long-term MA → inflation is accelerating
    2. Persistence = |corr(CPI_t, CPI_{t-1}, 5-day)|
       Absolute value of the 5-day rolling correlation between CPI and its 1-lag → measures CPI's "inertia"
       High autocorrelation → CPI trend is strong and unlikely to jump → the current direction signal is more reliable
    """
    cpi = inds["PMF0042"]
    signal = zero2nan(np.sign(ts_mean(cpi, 5) - ts_mean(cpi, 20)))
    persistence = np.abs(ts_corr_binary(cpi, ts_delay(cpi, 1), 5))
    result = 1 * signal * persistence
    result.name = "pmf_102135"
    return result


def factor_rates_2_pmf_003678(inds: dict[str, pd.Series]) -> pd.Series:
    """20-day correlation change between TIPS real-yield change and 1-period-lagged ETF holdings change (identity template).

    Composition template: identity
    Indicators used: PMF0002 (TIPS real yield), PMF0019 (ETF holdings)

    Logic:
    1. Daily diff of TIPS
    2. Daily diff of ETF holdings, lagged by 1 period
    3. 20-day rolling Pearson correlation between the two
    4. Take the 20-day change of the correlation → directly the factor value (no additional transformation)

    Economic meaning: when the co-movement between real-yield change and ETF
    holdings change shifts, it may foreshadow a shift in the market's
    pricing logic.
    """
    tips_diff = inds["PMF0002"] - ts_delay(inds["PMF0002"], 1)
    etf_diff_lag = ts_delay(inds["PMF0019"] - ts_delay(inds["PMF0019"], 1), 1)
    corr_20d = ts_corr_binary(tips_diff, etf_diff_lag, 20)
    result = 1 * (corr_20d - ts_delay(corr_20d, 20))
    result.name = "pmf_003678"
    return result


def factor_rates_3_pmf_118259(inds: dict[str, pd.Series]) -> pd.Series:
    """Direction from CPI 20-day MA vs 60-day MA × CPI's position within its Bollinger band.

    Composition template: direction_x_position
    Indicators used: PMF0042 (US core CPI YoY)

    Logic:
    1. Direction = sign(MA20 - MA60): medium-term CPI trend direction
    2. Position = (CPI - lower band) / (band width) - 0.5
       - Lower band = MA5 - 1×STD5
       - Band width = 2×1×STD5
       - Resulting range approximately [-0.5, 0.5]:
         0.5 → CPI at the upper band (relatively high vs history)
         -0.5 → CPI at the lower band (relatively low vs history)
         0 → CPI at the middle band (near the mean)
    3. direction × position = factor value
       Inflation accelerating + CPI at highs → strong bullish signal
       Inflation decelerating + CPI at lows → strong bearish signal
    """
    cpi = inds["PMF0042"]
    direction = zero2nan(np.sign(ts_mean(cpi, 20) - ts_mean(cpi, 60)))
    cpi_std5 = ts_std(cpi, 5)
    lower_band = ts_mean(cpi, 5) - 1 * cpi_std5           # Bollinger lower band (k=1)
    position = (cpi - lower_band) / zero2nan(2 * 1 * cpi_std5) - 0.5  # Normalized to [-0.5, 0.5]
    result = 1 * direction * position
    result.name = "pmf_118259"
    return result


# ---- Theme 5: Risk / Volatility / Correlation ----

def factor_risk_1_pmf_051056(inds: dict[str, pd.Series]) -> pd.Series:
    """GPR 5-day trend direction × 20-day CPI return intensity.

    Composition template: direction_x_intensity
    Indicators used: PMF0007 (GPR), PMF0042 (CPI)

    Logic:
    1. 5-day MA of GPR vs its value 5 days ago; take the sign → direction (is geopolitical risk escalating or cooling?)
    2. |CPI / CPI_{t-20} - 1| → absolute value of 20-day CPI return → intensity
       Measures how much inflation has changed over the past 20 days
    3. direction × intensity = factor value
       Rising geopolitical risk + sharp inflation change → strong safe-haven demand
    """
    gpr_ma5 = ts_mean(inds["PMF0007"], 5)
    direction = zero2nan(np.sign(gpr_ma5 - ts_delay(gpr_ma5, 5)))
    intensity = np.abs(inds["PMF0042"] / zero2nan(ts_delay(inds["PMF0042"], 20)) - 1)
    result = 1 * direction * intensity
    result.name = "pmf_051056"
    return result


def factor_risk_2_pmf_055491(inds: dict[str, pd.Series]) -> pd.Series:
    """Option implied volatility 20-day trend direction × 5-day mean intensity of (CPI daily diff / log|SPDR cash flow|).

    Composition template: direction_x_intensity
    Indicators used: PMF0018 (option volatility), PMF0042 (CPI), PMF0036 (SPDR cash flow)

    Logic:
    1. 20-day MA of option volatility vs its value 20 days ago; take the sign → direction
       Rising volatility → the market is "afraid"
    2. Intensity = 5-day mean of (|CPI daily change| / log|SPDR cash flow|)
       - CPI daily change → instantaneous inflation fluctuation
       - log|SPDR cash flow| → log of the cash-flow magnitude (log compresses the scale)
       - Ratio → inflation change per unit of cash-flow magnitude
    """
    vol_ma20 = ts_mean(inds["PMF0018"], 20)
    direction = zero2nan(np.sign(vol_ma20 - ts_delay(vol_ma20, 20)))
    cpi_diff = np.abs(inds["PMF0042"] - ts_delay(inds["PMF0042"], 1))
    log_spdr_cash = np.abs(np.log(zero2nan(np.abs(inds["PMF0036"]))))
    intensity = ts_mean(cpi_diff / zero2nan(log_spdr_cash), 5)
    result = 1 * direction * intensity
    result.name = "pmf_055491"
    return result


def factor_risk_3_pmf_066409(inds: dict[str, pd.Series]) -> pd.Series:
    """Direction from GPR 20-day MA vs 60-day MA × 5-day mean intensity of |term spread| / log|SPDR flow|.

    Composition template: direction_x_intensity
    Indicators used: PMF0007 (GPR), PMF0013 (10Y-2Y term spread), PMF0031 (SPDR flow)

    Logic:
    1. Sign of (MA20 of GPR - MA60 of GPR) → medium-to-long-term geopolitical risk trend
       (Longer than pmf_051056's 5-day window, capturing slower trends)
    2. Intensity = 5-day mean of (|term spread| / log|SPDR flow|)
       - Absolute value of term spread → steepness/inversion degree of the yield curve
       - log|SPDR flow| → log of gold ETF fund flow
       - Ratio → term-spread magnitude per unit of ETF fund flow
    """
    direction = zero2nan(
        np.sign(ts_mean(inds["PMF0007"], 20) - ts_mean(inds["PMF0007"], 60))
    )
    log_spdr = np.abs(np.log(zero2nan(np.abs(inds["PMF0031"]))))
    intensity = ts_mean(np.abs(inds["PMF0013"]) / zero2nan(log_spdr), 5)
    result = 1 * direction * intensity
    result.name = "pmf_066409"
    return result


# ======================================================================
# Part 5: Factor registry and runner
# ======================================================================

# Mapping from factor name → computation function.
# build_factors() iterates this table to compute factors in batch.
TOP15_FACTORS: dict[str, Callable[[dict[str, pd.Series]], pd.Series]] = {
    "pmf_128453": factor_dollar_1_pmf_128453,
    "pmf_011095": factor_dollar_2_pmf_011095,
    "pmf_069545": factor_dollar_3_pmf_069545,
    "pmf_051113": factor_goldprice_1_pmf_051113,
    "pmf_127111": factor_goldprice_2_pmf_127111,
    "pmf_125861": factor_goldprice_3_pmf_125861,
    "pmf_131810": factor_etf_1_pmf_131810,
    "pmf_104203": factor_etf_2_pmf_104203,
    "pmf_058341": factor_etf_3_pmf_058341,
    "pmf_102135": factor_rates_1_pmf_102135,
    "pmf_003678": factor_rates_2_pmf_003678,
    "pmf_118259": factor_rates_3_pmf_118259,
    "pmf_051056": factor_risk_1_pmf_051056,
    "pmf_055491": factor_risk_2_pmf_055491,
    "pmf_066409": factor_risk_3_pmf_066409,
}


def run_dry_check(indicator_root: Path, pmf_map: dict[str, str]) -> int:
    """Dry-run check: verify each PMF file exists and is readable. Returns exit code (0=pass, 1=problem).

    For each PMF code:
    - Check whether the file exists
    - Attempt to load it and print the observation count and date range
    - Mark load failures as MISSING or PARSE ERROR
    """
    print("=" * 70)
    print(f"Dry-run check: verifying {len(pmf_map)} PMF mappings, root directory:")
    print(f"  {indicator_root}")
    print("=" * 70)

    missing = []
    for pmf_code, rel_path in pmf_map.items():
        full_path = indicator_root / rel_path
        if not full_path.exists():
            print(f"  x {pmf_code:<8s} FILE MISSING  {rel_path}")
            missing.append(pmf_code)
            continue
        try:
            series = load_indicator(pmf_code, indicator_root, pmf_map)
            n_obs = int(series.notna().sum())
            date_min = series.index.min().date()
            date_max = series.index.max().date()
            print(
                f"  v {pmf_code:<8s} {n_obs:>5d} obs  {date_min} -> {date_max}  "
                f"{rel_path}"
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"  x {pmf_code:<8s} PARSE ERROR  {rel_path}: {exc}")
            missing.append(pmf_code)

    print("=" * 70)
    if missing:
        print(f"FAILED: {len(missing)} mappings failed to parse: {missing}")
        return 1
    print(f"PASSED: all {len(pmf_map)} PMF mappings verified.")
    return 0


def build_factors(
    indicator_root: Path,
    pmf_map: dict[str, str],
    factor_names: list[str] | None = None,
) -> pd.DataFrame:
    """Load indicators, compute the specified factors, and return a wide-format DataFrame.

    Steps:
    1. Load all 13 indicators into a dictionary
    2. Iterate over the factor list to compute (default: all 15)
    3. Call each factor's computation function and collect the results
    4. Horizontally concatenate all factor Series into a single DataFrame
    """
    print("=" * 70)
    print("Step 1: load indicators")
    print("=" * 70)
    inds = load_all_indicators(indicator_root, pmf_map)  # Dictionary of all indicator values

    factors_to_run = factor_names or list(TOP15_FACTORS.keys())

    print()
    print("=" * 70)
    print(f"Step 2: compute {len(factors_to_run)} factors")
    print("=" * 70)

    frames: list[pd.Series] = []
    for i, factor_name in enumerate(factors_to_run, start=1):
        if factor_name not in TOP15_FACTORS:
            logger.warning("Unknown factor name: %s (skipped)", factor_name)
            continue
        factor_fn = TOP15_FACTORS[factor_name]
        try:
            series = factor_fn(inds)
            frames.append(series)
            print(
                f"  [{i:2d}/{len(factors_to_run):2d}] {factor_name:<20s}  "
                f"-> {series.notna().sum()} non-null values"
            )
        except (KeyError, NotImplementedError) as exc:
            logger.warning("Skipping %s: %s", factor_name, exc)

    if not frames:
        raise RuntimeError("No factor series were produced. Check PMF mappings and input files.")

    # Horizontally concatenate all factor Series and sort by date
    return pd.concat(frames, axis=1).sort_index()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only verify PMF file mappings; do not compute factors.",
    )
    parser.add_argument(
        "--factors",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Compute only the specified factor names (default: all 15).",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry: dispatch to a dry-run check or a full computation based on CLI args."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    # In dry-run mode, only verify file mappings and exit
    if args.dry_run:
        return run_dry_check(INDICATOR_ROOT, PMF_TO_FILENAME)

    # Full run: compute the factors
    factor_df = build_factors(INDICATOR_ROOT, PMF_TO_FILENAME, args.factors)

    # Truncate to the cutoff date
    factor_df = factor_df[factor_df.index <= CUTOFF_DATE]

    # Ensure the output directory exists and write the CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    factor_df.to_csv(OUTPUT_PATH)
    print()
    print(
        f"Saved {factor_df.shape[1]} factor columns x {factor_df.shape[0]} rows to "
        f"{OUTPUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
