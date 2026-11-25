"""Top-15 gold factor backtest engine (Sharpe-weighted portfolio).

Pipeline
--------
    factor wide table  +  Au9995 daily close  ->  per-factor NAV, portfolio NAV
                                                  +  vol-normalized HS300 benchmark

For each of the 15 factors and each of the 3 position-mapping styles below,
this script:

    1. Computes a full-sample Spearman IC to fix the trade direction (+1 or -1).
    2. Maps the raw factor value to a target position in [-1, +1] using ONE of
       - ``rolling_percentage``     rolling percentile -> linear map to [-1, 1]
       - ``rolling_zscore_tanh``    rolling z-score, tanh-squashed
       - ``rolling_zscore_clip``    rolling z-score, hard-clipped to [-1, 1]
    3. Sparsifies rebalances to every HOLDING_PERIOD (=10) trading days, then
       TWAP-smooths the resulting step function over POSITION_SMOOTH_WINDOW
       (=5) days. In a linear-cost model these two operations preserve total
       turnover while spreading each rebalance over 5 days - the smoothing
       becomes a design deposit for future non-linear cost models.
    4. Executes with a T+1 lag on the Au9995 simple return, deducts 3 bp per
       unit of turnover, compounds to NAV, and records the standard metrics.

The 15 single-factor backtests are then combined into a portfolio with
weights proportional to each factor's single-name Sharpe (``SHARPE_FLOOR``
applied as a soft cutoff). We deliberately combine the DIRECTIONAL raw
positions and post-process the sum ONCE, not the already-post-processed
per-factor positions - see notes on the ``postprocess_position`` function.

Benchmark
---------
CSI 300 (000300.SH) is loaded, its daily returns are scaled down to
``BENCHMARK_TARGET_ANN_VOL`` (5.5%, matching the main portfolio's realized
vol), and the scaled series is compounded to produce an apples-to-apples
buy-and-hold benchmark. Scaling at the return level (r' = r * scale) is
mandatory: geometrically scaling an already-built NAV curve preserves the
shape but does not preserve annualized volatility.

Outputs (all under ``OUTPUT_DIR``)
----------------------------------
    - IC_spearman.csv              full-sample Spearman IC + direction
    - single_factor_metrics.csv    15 factors x 3 position styles = 45 rows
    - portfolio_metrics.csv        6 rows (Sharpe-weighted vs equal-weighted x 3 styles)
    - portfolio_weights.csv        45 rows of per-factor weights per style
    - plots/{factor}.png           per-factor NAV, 3 styles overlaid + benchmark
    - plots/_portfolio_sharpe_weighted.png
    - plots/_portfolio_equal_weighted.png
"""

from __future__ import annotations

import logging
import os
import warnings
from math import sqrt
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

logger = logging.getLogger(__name__)


# ======================================================
# User configuration (adjust these paths and constants)
# ======================================================

# Factor wide table (rows = trading days, columns = factor codes)
FACTOR_PATH = os.environ.get(
    "FACTOR_CSV_PATH",
    "results/factor_values_top15.csv",
)

# Au9995 daily close (columns: date, price) - the strategy's traded asset
AU9995_PATH = os.environ.get(
    "AU9995_PATH",
    "data/aligned/daily/Au9995_close.csv",
)

# CSI 300 daily close (columns: date, close) - benchmark reference
HS300_PATH = os.environ.get(
    "HS300_PATH",
    "data/aligned/daily/000300_SH.csv",
)

# Target annualized volatility for the vol-normalized benchmark. Set to the
# realized annualized vol of the main portfolio (5.5%) so absolute-return /
# drawdown comparisons are apples-to-apples.
BENCHMARK_TARGET_ANN_VOL = 0.055

OUTPUT_DIR = os.environ.get("BACKTEST_OUTPUT_DIR", "results/backtest")

# Sample truncation
CUTOFF_DATE = "2023-12-31"
START_DATE: Optional[str] = None

# --- Backtest hyperparameters -----------------------------------------------
TRADING_DAYS_PER_YEAR = 252
ROLLING_WINDOW = 252            # rolling window for position mapping
MIN_PERIODS_ROLLING = 60        # min samples before mapping emits a value
COST_RATE = 0.0003              # 3 bp per unit of turnover, one-sided
EPSILON = 1e-8

# --- Portfolio construction knobs -------------------------------------------
# (1) Combine per-factor signals with weights proportional to single-name
#     Sharpe. Sharpe below SHARPE_FLOOR is clamped to zero, effectively
#     dropping the factor. 0.0 keeps all positive-Sharpe factors (Plan A).
SHARPE_FLOOR = 0.0
# (2) Rebalance every HOLDING_PERIOD trading days; between rebalances the
#     position is held constant via forward-fill. HOLDING_PERIOD=1 disables.
HOLDING_PERIOD = 10
# (3) After sparsifying, TWAP-smooth over POSITION_SMOOTH_WINDOW days. In a
#     linear cost model this preserves total turnover; it matters once the
#     cost model becomes non-linear (market impact).
POSITION_SMOOTH_WINDOW = 5

POSITION_TYPES = ["rolling_percentage", "rolling_zscore_tanh", "rolling_zscore_clip"]


# ======================================================
# Section 2: Data loading utilities
# ======================================================

def _read_csv_any_encoding(path: str, **kwargs) -> pd.DataFrame:
    """Try utf-8 / gbk / gb2312 / latin1 in order. Vendor CSVs come in many encodings."""
    for enc in ("utf-8", "gbk", "gb2312", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"Cannot decode file: {path}")


def load_factor_table(path: str) -> pd.DataFrame:
    """Load the factor wide table as a date-indexed DataFrame.

    Supports both the 3.0 CSV (native ISO dates) and the 2.0 XLSX
    (date column is an Excel serial number with origin 1899-12-30).
    """
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = _read_csv_any_encoding(path)

    df.columns = df.columns.str.strip()
    date_col = next((c for c in df.columns if c.lower() == "date"), df.columns[0])

    date_series = df[date_col]
    if pd.api.types.is_numeric_dtype(date_series):
        df[date_col] = pd.to_datetime(date_series, unit="D", origin="1899-12-30")
    else:
        df[date_col] = pd.to_datetime(date_series, errors="coerce")

    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return df[~df.index.duplicated(keep="last")]


def load_au9995(path: str) -> pd.Series:
    """Load Au9995 close prices; returns a date-indexed Series named 'Au9995'."""
    df = _read_csv_any_encoding(path)
    df.columns = df.columns.str.strip()
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    price_col = next((c for c in df.columns if c.lower() == "price"), None)
    if date_col is None or price_col is None:
        raise ValueError(f"Au9995 file missing date or price column: {path}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return pd.to_numeric(df[price_col], errors="coerce").rename("Au9995")


def load_hs300(path: str) -> pd.Series:
    """Load CSI 300 close prices (columns: date, close). Named 'HS300'."""
    df = _read_csv_any_encoding(path)
    df.columns = df.columns.str.strip()
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    close_col = next((c for c in df.columns if c.lower() == "close"), None)
    if date_col is None or close_col is None:
        raise ValueError(f"CSI 300 file missing date or close column: {path}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return pd.to_numeric(df[close_col], errors="coerce").rename("HS300")


# ======================================================
# Section 3: Position mappings (raw factor -> position in [-1, +1])
# ======================================================
# All three functions produce an unsigned position magnitude in [-1, +1].
# The trade direction (sign of IC) is applied by the caller as a scalar.

def position_rolling_percentage(x: pd.Series) -> pd.Series:
    """Rolling percentile rank of the current value within a WINDOW-day history.

    Percentile p in [0, 1] -> position 2p - 1 in [-1, +1].
    The current point is excluded from its own history (window[:-1]) to prevent
    look-ahead bias. Extreme values collapse cleanly to +/- 1 without further
    clipping.
    """
    def _pct_rank(window: np.ndarray) -> float:
        return float((window[:-1] < window[-1]).mean())

    pct = x.rolling(ROLLING_WINDOW + 1, min_periods=MIN_PERIODS_ROLLING + 1).apply(
        _pct_rank, raw=True,
    )
    return pct * 2 - 1


def position_rolling_zscore_tanh(x: pd.Series) -> pd.Series:
    """Rolling z-score squashed through tanh - smoothly bounded in (-1, +1).

    z = (x - rolling_mean) / rolling_std
    Extreme z-scores taper toward +/- 1 gradually instead of being clipped.
    """
    mean = x.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS_ROLLING).mean()
    std = x.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS_ROLLING).std()
    z = (x - mean) / std.where(std > EPSILON, np.nan)
    return np.tanh(z)


def position_rolling_zscore_clip(x: pd.Series) -> pd.Series:
    """Rolling z-score hard-clipped to [-1, +1] - a harder-edged tanh."""
    mean = x.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS_ROLLING).mean()
    std = x.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS_ROLLING).std()
    z = (x - mean) / std.where(std > EPSILON, np.nan)
    return z.clip(-1.0, 1.0)


POSITION_FUNCS = {
    "rolling_percentage":  position_rolling_percentage,
    "rolling_zscore_tanh": position_rolling_zscore_tanh,
    "rolling_zscore_clip": position_rolling_zscore_clip,
}


# ======================================================
# Section 3b: Position post-processing (holding period + TWAP smoothing)
# ======================================================

def apply_holding_period(pos: pd.Series, period: int) -> pd.Series:
    """Sparsify a position series so rebalances only occur every `period` days.

    period=1  -> identity (rebalance every day).
    period=10 -> only days 0, 10, 20, ... update; the rest are ffilled.

    Effect: turnover frequency drops to 1/period, cutting linear costs
    proportionally.
    """
    if period <= 1:
        return pos
    valid_idx = pos.dropna().index
    if len(valid_idx) == 0:
        return pos
    rebal_dates = set(valid_idx[::period])
    out = pos.copy()
    non_rebal = ~out.index.isin(rebal_dates)
    out[non_rebal] = np.nan
    return out.ffill()


def smooth_position(pos: pd.Series, window: int) -> pd.Series:
    """Apply a `window`-day rolling-mean smoother to a position series.

    window=1 -> identity.
    window=5 -> the position on any given day is the mean of the last 5 days.

    When it follows `apply_holding_period` this is a TWAP execution overlay:
    each rebalance is spread linearly across the next `window` days, dampening
    single-day position jumps.
    """
    if window <= 1:
        return pos
    return pos.rolling(window, min_periods=1).mean()


def postprocess_position(pos: pd.Series) -> pd.Series:
    """Canonical post-processing pipeline for any position series.

    Order matters: `apply_holding_period` first produces a step function, then
    `smooth_position` linearly interpolates each step over the TWAP window.
    Reversing the order changes the behavior from "TWAP execution overlay" to
    "smoothed signal followed by sparsification" and drops Sharpe by ~0.2.

    Applied identically to single factors and to the combined portfolio so
    that the two are directly comparable.
    """
    pos = apply_holding_period(pos, HOLDING_PERIOD)
    pos = smooth_position(pos, POSITION_SMOOTH_WINDOW)
    return pos


# ======================================================
# Section 4: Full-sample IC (used only to fix trade direction)
# ======================================================

def compute_ic_direction(
    factor_df: pd.DataFrame, forward_return: pd.Series,
) -> pd.DataFrame:
    """Compute full-sample Spearman IC for each factor and derive its sign.

    The sign of the IC decides whether we go long or short the factor's
    exposure in the single-factor backtest. Portfolio weights are computed
    separately (see main step 4) from realized single-factor Sharpe, not IC.

    Returns
    -------
    DataFrame indexed by factor with columns ``ic_spearman``, ``direction``,
    ``n_obs``.
    """
    records = []
    for factor_name in factor_df.columns:
        aligned = pd.concat(
            [factor_df[factor_name], forward_return], axis=1,
        ).dropna()
        if len(aligned) < 30:
            ic = np.nan
        else:
            ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
        direction = int(np.sign(ic)) if not np.isnan(ic) else 0
        records.append({
            "factor": factor_name,
            "ic_spearman": ic,
            "direction": direction,
            "n_obs": len(aligned),
        })
    return pd.DataFrame(records).set_index("factor")


# ======================================================
# Section 5: Single-run backtest (position series -> metrics)
# ======================================================

def backtest_position(
    position: pd.Series, simple_return: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    """Backtest one position series against Au9995 daily simple return.

    Order of operations
    -------------------
        1. T+1 execution:   pos_lag_t = position_{t-1}
        2. gross_t = pos_lag_t * simple_return_t
        3. turnover_t = |pos_lag_t - pos_lag_{t-1}|
        4. net_t = gross_t - turnover_t * COST_RATE
        5. NAV_t = product(1 + net_s)   -- compounded (not additive), so a
                                            drawdown never drives NAV negative
        6. Standard metrics: ann. return, ann. vol, Sharpe, MDD, Calmar, win%.

    Returns (net_return, nav, turnover_series, metrics_dict).
    """
    idx = position.dropna().index.intersection(simple_return.dropna().index)
    pos = position.reindex(idx)
    ret = simple_return.reindex(idx)

    pos_lag = pos.shift(1)
    turnover = pos_lag.diff().abs().fillna(0.0)
    gross = pos_lag * ret
    net = gross - turnover * COST_RATE

    valid_net = net.dropna()
    total_days = len(valid_net)

    if total_days == 0:
        nan_metrics = {k: np.nan for k in (
            "ann_return", "ann_turnover", "ann_vol",
            "Sharpe", "max_drawdown", "Calmar", "win_rate",
        )}
        return net, pd.Series(dtype=float), turnover, nan_metrics

    nav = (1.0 + net.fillna(0.0)).cumprod()

    ann_return = valid_net.mean() * TRADING_DAYS_PER_YEAR
    ann_turnover = turnover.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = valid_net.std() * sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > EPSILON else np.nan

    rolling_max = nav.cummax()
    max_dd = ((rolling_max - nav) / rolling_max).max()
    calmar = ann_return / max_dd if max_dd > EPSILON else np.nan

    non_zero = valid_net[valid_net != 0]
    win_rate = float((non_zero > 0).mean()) if len(non_zero) > 0 else np.nan

    metrics = {
        "ann_return":   round(float(ann_return), 4),
        "ann_turnover": round(float(ann_turnover), 4),
        "ann_vol":      round(float(ann_vol), 4),
        "Sharpe":       round(float(sharpe), 3) if not np.isnan(sharpe) else np.nan,
        "max_drawdown": round(float(max_dd), 4),
        "Calmar":       round(float(calmar), 3) if not np.isnan(calmar) else np.nan,
        "win_rate":     round(float(win_rate), 4) if not np.isnan(win_rate) else np.nan,
    }
    return net, nav, turnover, metrics


# ======================================================
# Section 6: Plotting - NAV + rolling Sharpe per factor
# ======================================================

def plot_factor_navs(
    factor_name: str,
    navs: dict[str, pd.Series],
    bh_nav: pd.Series,
    metrics_by_type: dict[str, dict],
    save_path: str,
) -> None:
    """Two-panel figure: top = NAV curves (3 styles + benchmark), bottom = rolling 252-day Sharpe."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"hspace": 0.5})

    colors = {
        "rolling_percentage":  "steelblue",
        "rolling_zscore_tanh": "darkorange",
        "rolling_zscore_clip": "seagreen",
    }
    labels = {
        "rolling_percentage":  "Rolling percentile",
        "rolling_zscore_tanh": "Z-Tanh",
        "rolling_zscore_clip": "Z-Clip",
    }

    ax1 = axes[0]
    plot_idx_union = None
    for ptype, nav in navs.items():
        nav_valid = nav.dropna()
        if len(nav_valid) == 0:
            continue
        ax1.plot(nav_valid.index, nav_valid.values,
                 color=colors[ptype], linewidth=1.1, label=labels[ptype], alpha=0.9)
        plot_idx_union = (
            nav_valid.index if plot_idx_union is None
            else plot_idx_union.union(nav_valid.index)
        )

    if plot_idx_union is not None:
        bh_seg = bh_nav.reindex(plot_idx_union).dropna()
        if len(bh_seg) > 0:
            bh_seg = bh_seg / bh_seg.iloc[0]
            ax1.plot(bh_seg.index, bh_seg.values, color="gray",
                     linewidth=1.0, linestyle="--", alpha=0.7,
                     label=f"CSI 300 (vol-normalized to {BENCHMARK_TARGET_ANN_VOL:.1%})")

    ax1.axhline(1.0, color="black", linewidth=0.5, linestyle=":", alpha=0.6)

    title_lines = [factor_name]
    for ptype in POSITION_TYPES:
        m = metrics_by_type.get(ptype, {})
        if m and not pd.isna(m.get("Sharpe", np.nan)):
            title_lines.append(
                f"{labels[ptype]}: "
                f"ann={m['ann_return']:.2%}  "
                f"Sharpe={m['Sharpe']}  "
                f"MDD={m['max_drawdown']:.2%}"
            )
    ax1.set_title("\n".join(title_lines), fontsize=9, fontweight="bold")
    ax1.set_ylabel("NAV")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]

    def _rolling_sharpe(x: np.ndarray) -> float:
        s = x.std()
        return x.mean() / s * sqrt(TRADING_DAYS_PER_YEAR) if s > EPSILON else np.nan

    for ptype, nav in navs.items():
        net_approx = nav.diff().dropna()
        if len(net_approx) == 0:
            continue
        rs = net_approx.rolling(TRADING_DAYS_PER_YEAR, min_periods=126).apply(
            _rolling_sharpe, raw=True,
        )
        ax2.plot(rs.index, rs.values, color=colors[ptype],
                 linewidth=1.0, label=labels[ptype], alpha=0.9)

    ax2.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax2.set_title("Rolling 252-day Sharpe", fontsize=10)
    ax2.set_ylabel("Sharpe")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================
# Section 7: Main
# ======================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 70)
    print("Step 1: Load data")
    print("=" * 70)

    factor_df = load_factor_table(FACTOR_PATH)
    factor_df = factor_df[factor_df.index <= CUTOFF_DATE]
    if START_DATE:
        factor_df = factor_df[factor_df.index >= START_DATE]
    print(f"  Factor table: {factor_df.shape[0]} rows x {factor_df.shape[1]} cols")

    au_price = load_au9995(AU9995_PATH)
    au_price = au_price[au_price.index <= CUTOFF_DATE]
    if START_DATE:
        au_price = au_price[au_price.index >= START_DATE]
    print(f"  Au9995: {len(au_price)} rows")

    # Simple daily return: r_t = (P_t - P_{t-1}) / P_{t-1}
    # Combined with pos.shift(1) in backtest_position, this realizes
    # "yesterday's signal earns today's return" - i.e. T+1 execution.
    simple_ret = au_price.pct_change().rename("ret")

    hs300_price = load_hs300(HS300_PATH)
    hs300_price = hs300_price[hs300_price.index <= CUTOFF_DATE]
    if START_DATE:
        hs300_price = hs300_price[hs300_price.index >= START_DATE]

    # Vol normalization: scale HS300 daily returns to BENCHMARK_TARGET_ANN_VOL,
    # then compound. Scale at return level (r' = r * scale), NOT after building
    # NAV (that would preserve shape but not annualized vol).
    hs300_ret = hs300_price.pct_change()
    raw_ann_vol = hs300_ret.std() * sqrt(TRADING_DAYS_PER_YEAR)
    vol_scale = BENCHMARK_TARGET_ANN_VOL / raw_ann_vol
    hs300_scaled_ret = hs300_ret * vol_scale
    bh_nav = (1.0 + hs300_scaled_ret.fillna(0.0)).cumprod()
    print(f"  CSI 300: {len(hs300_price)} rows  raw ann vol={raw_ann_vol:.2%} "
          f"-> scale={vol_scale:.3f} -> target vol={BENCHMARK_TARGET_ANN_VOL:.2%}")

    print("\n" + "=" * 70)
    print("Step 2: Full-sample Spearman IC -> direction for single-factor backtest")
    print("=" * 70)

    ic_df = compute_ic_direction(factor_df, simple_ret)
    ic_df.to_csv(os.path.join(OUTPUT_DIR, "IC_spearman.csv"), encoding="utf-8-sig")
    print(ic_df.round(4).to_string())

    print("\n" + "=" * 70)
    print(f"Step 3: Single-factor backtest (15 factors x 3 mappings; "
          f"HOLDING={HOLDING_PERIOD}d, SMOOTH={POSITION_SMOOTH_WINDOW}d)")
    print("=" * 70)

    single_records = []
    # Cache raw_mapped * direction (BEFORE post-processing). The portfolio
    # combines these and post-processes ONCE at the sum level, avoiding the
    # trap of "smooth each factor + smooth the sum" (double-smoothing that
    # kills Sharpe by ~0.6 in practice).
    directional_positions: dict[str, dict[str, pd.Series]] = {
        ptype: {} for ptype in POSITION_TYPES
    }
    sharpes: dict[str, dict[str, float]] = {ptype: {} for ptype in POSITION_TYPES}

    for factor_name in factor_df.columns:
        direction = ic_df.loc[factor_name, "direction"] or 1
        raw = factor_df[factor_name]

        navs_this_factor: dict[str, pd.Series] = {}
        metrics_this_factor: dict[str, dict] = {}

        for ptype in POSITION_TYPES:
            pos_mapped = POSITION_FUNCS[ptype](raw)
            pos_dir = pos_mapped * direction
            pos = postprocess_position(pos_dir)

            _, nav, _, metrics = backtest_position(pos, simple_ret)

            directional_positions[ptype][factor_name] = pos_dir
            sharpes[ptype][factor_name] = (
                metrics["Sharpe"] if not pd.isna(metrics["Sharpe"]) else 0.0
            )
            navs_this_factor[ptype] = nav
            metrics_this_factor[ptype] = metrics

            single_records.append({
                "factor":        factor_name,
                "position_type": ptype,
                "direction":     int(direction),
                **metrics,
            })

        safe_name = factor_name.replace("/", "_").replace(" ", "_")
        plot_factor_navs(
            factor_name=factor_name,
            navs=navs_this_factor,
            bh_nav=bh_nav,
            metrics_by_type=metrics_this_factor,
            save_path=os.path.join(plots_dir, f"{safe_name}.png"),
        )

        pct_m = metrics_this_factor["rolling_percentage"]
        print(
            f"  {factor_name:<12s} dir={int(direction):+d}  "
            f"[pct] ann={pct_m['ann_return']:>7.2%} Sharpe={pct_m['Sharpe']}  "
            f"MDD={pct_m['max_drawdown']:>6.2%}"
        )

    single_df = pd.DataFrame(single_records)
    single_csv = os.path.join(OUTPUT_DIR, "single_factor_metrics.csv")
    single_df.to_csv(single_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Single-factor metrics saved: {single_csv}  ({len(single_df)} rows)")

    print("\n" + "=" * 70)
    print(f"Step 4: Portfolio backtest - Sharpe-weighted (floor={SHARPE_FLOOR}) "
          f"vs equal-weighted baseline")
    print("=" * 70)

    portfolio_records = []
    sw_navs: dict[str, pd.Series] = {}
    ew_navs: dict[str, pd.Series] = {}
    sw_metrics_map: dict[str, dict] = {}
    ew_metrics_map: dict[str, dict] = {}
    weight_records = []

    for ptype in POSITION_TYPES:
        pos_matrix = pd.DataFrame(directional_positions[ptype])
        sharpe_series = pd.Series(sharpes[ptype])

        # Sharpe-weighted: weight_i = max(Sharpe_i - floor, 0) / sum(...)
        raw_w = (sharpe_series - SHARPE_FLOOR).clip(lower=0.0)
        total = raw_w.sum()
        weights = raw_w / total if total > EPSILON else pd.Series(0.0, index=sharpe_series.index)

        n_selected = int((weights > 0).sum())

        pos_sw = postprocess_position(
            pos_matrix.mul(weights, axis=1).sum(axis=1, min_count=1)
        )
        _, nav_sw, _, m_sw = backtest_position(pos_sw, simple_ret)
        sw_navs[ptype] = nav_sw
        sw_metrics_map[ptype] = m_sw
        portfolio_records.append({
            "portfolio":     "sharpe_weighted",
            "position_type": ptype,
            "n_selected":    n_selected,
            **m_sw,
        })

        for factor_name in weights.index:
            weight_records.append({
                "position_type": ptype,
                "factor":        factor_name,
                "single_sharpe": round(float(sharpe_series[factor_name]), 3),
                "weight":        round(float(weights[factor_name]), 4),
                "selected":      bool(weights[factor_name] > 0),
            })

        # Equal-weighted baseline (all 15 factors, no filtering)
        pos_ew = postprocess_position(pos_matrix.mean(axis=1))
        _, nav_ew, _, m_ew = backtest_position(pos_ew, simple_ret)
        ew_navs[ptype] = nav_ew
        ew_metrics_map[ptype] = m_ew
        portfolio_records.append({
            "portfolio":     "equal_weighted",
            "position_type": ptype,
            "n_selected":    len(pos_matrix.columns),
            **m_ew,
        })

        print(
            f"  [{ptype:<22s}]  "
            f"SW(n={n_selected}) Sharpe={m_sw['Sharpe']:>6} ann={m_sw['ann_return']:>7.2%}  |  "
            f"EW Sharpe={m_ew['Sharpe']:>6} ann={m_ew['ann_return']:>7.2%}"
        )

    port_df = pd.DataFrame(portfolio_records)
    port_csv = os.path.join(OUTPUT_DIR, "portfolio_metrics.csv")
    port_df.to_csv(port_csv, index=False, encoding="utf-8-sig")
    print(f"\n  Portfolio metrics saved: {port_csv}  ({len(port_df)} rows)")

    weights_df = pd.DataFrame(weight_records)
    weights_csv = os.path.join(OUTPUT_DIR, "portfolio_weights.csv")
    weights_df.to_csv(weights_csv, index=False, encoding="utf-8-sig")
    print(f"  Portfolio weights saved: {weights_csv}  ({len(weights_df)} rows)")

    plot_factor_navs(
        factor_name="Sharpe-weighted portfolio",
        navs=sw_navs,
        bh_nav=bh_nav,
        metrics_by_type=sw_metrics_map,
        save_path=os.path.join(plots_dir, "_portfolio_sharpe_weighted.png"),
    )
    plot_factor_navs(
        factor_name="Equal-weighted portfolio (15 factors, baseline)",
        navs=ew_navs,
        bh_nav=bh_nav,
        metrics_by_type=ew_metrics_map,
        save_path=os.path.join(plots_dir, "_portfolio_equal_weighted.png"),
    )

    print("\n" + "=" * 70)
    print(f"Done. All outputs in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
