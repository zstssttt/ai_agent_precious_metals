"""Out-of-sample test of the 6 surviving gold factors (2024-01-02 -> 2025-11-28).

Protocol
--------
The in-sample study (2015-01 -> 2023-12, see ``../README.md``) selected 6 of
15 factors and one position-mapping style. This script re-runs that exact
configuration on data the selection never saw, with **every fitted quantity
frozen at its in-sample value**:

    frozen from ../results/backtest_IC_spearman.csv  ->  trade direction (sign of IC)
    frozen from ../results/portfolio_weights.csv     ->  Sharpe-proportional weights
    frozen constants                                 ->  rolling_percentage mapping,
                                                         HOLDING_PERIOD=10, SMOOTH=5,
                                                         T+1 lag, 3 bp linear cost

Nothing is re-estimated on 2024-2025 data. The only thing this script decides
is *when* the evaluation window starts.

Warm-up handling
----------------
Factor formulas and the 252-day rolling percentile mapping are evaluated on
the FULL history (1990-12-19 -> 2025-11-28) and only the resulting daily net
return series is sliced to the evaluation window. Both operations look
strictly backward, so this introduces no look-ahead bias, and it means all
462 OOS trading days are usable instead of burning the first 60 on the
rolling window's ``min_periods``.

The T+1 lag is likewise applied before slicing, so the first OOS day earns
the return on the position held at the 2023-12-29 close - the window boundary
is not a free reset to flat.

Data cutoff: why 2025-11-28
---------------------------
Au9995 (the only traded asset) has real closes through 2025-11-28. Four
indicator files (COMEX gold, DXY, 10Y-2Y spread, total gold ETF holdings)
carry dates out to 2026-12-31, but their last genuine print is also
2025-11-28 - the remaining 266 rows are a flat forward-fill produced by the
trading-calendar aligner. Cutting at 2025-11-28 excludes all of them.

Outputs (under ``OOS/results/``)
--------------------------------
    oos_factor_values_6.csv        the 6 factors, full history through cutoff
    oos_single_factor_metrics.csv  6 rows, frozen direction, OOS window only
    oos_portfolio_metrics.csv      portfolio + benchmark rows, IS vs OOS
    oos_nav_daily.csv              daily NAV / net return / position / turnover
    oos_monthly_returns.csv        month-by-month net return of the portfolio
    plots/oos_portfolio_nav.png    OOS NAV vs benchmarks
    plots/oos_is_vs_oos_nav.png    continuous NAV with the IS/OOS split marked

Usage
-----
    export GOLD_FACTOR_ALIGNED_ROOT="/path/to/补齐后数据/补齐后数据"
    export AU9995_PATH="$GOLD_FACTOR_ALIGNED_ROOT/补齐后日频数据2.0/收盘价_Au9995_S0035818.csv"
    export HS300_PATH="/path/to/000300.SH.csv"
    python OOS/run_oos.py
"""

from __future__ import annotations

import os
import sys
import warnings
from math import sqrt
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# --- Reuse the in-sample engine verbatim -------------------------------------
# Importing rather than copy-pasting guarantees the OOS run uses byte-identical
# factor formulas, position mapping and post-processing.
OOS_DIR = Path(__file__).resolve().parent
REPO_ROOT = OOS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from src.factors.build_factors import (          # noqa: E402
    PMF_TO_FILENAME,
    TOP15_FACTORS,
    load_all_indicators,
)
from src.backtest.position_mapping import (      # noqa: E402
    COST_RATE,
    EPSILON,
    TRADING_DAYS_PER_YEAR,
    load_au9995,
    load_hs300,
    position_rolling_percentage,
    postprocess_position,
)


# ======================================================
# Configuration
# ======================================================

INDICATOR_ROOT = Path(os.environ.get("GOLD_FACTOR_ALIGNED_ROOT", "data/aligned"))
AU9995_PATH = os.environ.get("AU9995_PATH", "data/aligned/daily/Au9995_close.csv")
HS300_PATH = os.environ.get("HS300_PATH", "data/aligned/daily/000300_SH.csv")

OUTPUT_DIR = Path(os.environ.get("OOS_OUTPUT_DIR", OOS_DIR / "results"))
PLOTS_DIR = OUTPUT_DIR / "plots"

# In-sample artifacts the frozen parameters are read from.
IS_IC_PATH = REPO_ROOT / "results" / "backtest_IC_spearman.csv"
IS_WEIGHTS_PATH = REPO_ROOT / "results" / "portfolio_weights.csv"

# In-sample boundaries. NOTE: ``position_mapping.py`` leaves ``START_DATE =
# None``, so the headline in-sample Sharpe of 0.939 in ``../README.md`` is
# actually measured over 2005-01-06 -> 2023-12-29 (4,614 days), not over the
# "2015-01 -> 2023-12" the README labels it with. Both slices are reported
# below so the in-sample baseline for this OOS comparison is unambiguous:
#     full     2005-01-06 -> 2023-12-29   Sharpe 0.939  (the README number)
#     2015+    2015-01-02 -> 2023-12-29   Sharpe 0.901
IS_FULL_START = None            # no lower bound - Au9995 itself starts 2005-01-04
IS_2015_START = "2015-01-01"
IS_END = "2023-12-31"

# Evaluation window. OOS_END is dictated by the last real Au9995 close.
OOS_START = "2024-01-01"
OOS_END = "2025-11-28"

PERIODS = {
    "in_sample_full":     (IS_FULL_START, IS_END),
    "in_sample_2015plus": (IS_2015_START, IS_END),
    "out_of_sample":      (OOS_START, OOS_END),
}

# The winning configuration from the in-sample study.
POSITION_TYPE = "rolling_percentage"
SELECTED_FACTORS = [
    "pmf_128453",   # Dollar   : d(corr(DXY, SPDR flow)) / GPR
    "pmf_125861",   # Gold px  : d(corr(COMEX, SPDR flow)) / GPR
    "pmf_131810",   # ETF      : d(corr(ETF holdings, 10Y yield)) / GPR
    "pmf_051056",   # Risk     : sign(GPR trend) x |CPI 20d change|
    "pmf_055491",   # Risk     : sign(IV trend) x CPI-per-log-flow
    "pmf_066409",   # Risk     : sign(GPR MA cross) x |term spread| / log|flow|
]


# ======================================================
# Frozen in-sample parameters
# ======================================================

def load_frozen_direction() -> pd.Series:
    """Read the in-sample full-sample Spearman IC sign for the 6 factors.

    This is the *only* place the trade direction comes from. Recomputing IC on
    2024-2025 data would leak the OOS period's realized returns into the
    signal and the test would no longer be out-of-sample.
    """
    ic = pd.read_csv(IS_IC_PATH, encoding="utf-8-sig")
    ic.columns = ic.columns.str.strip()
    ic = ic.set_index("factor")
    missing = [f for f in SELECTED_FACTORS if f not in ic.index]
    if missing:
        raise KeyError(f"{IS_IC_PATH.name} is missing factors: {missing}")
    return ic.loc[SELECTED_FACTORS, "direction"].astype(int)


def load_frozen_weights() -> pd.Series:
    """Read the in-sample Sharpe-proportional weights for the winning mapping.

    The in-sample weights already sum to 1.0 across exactly these 6 factors
    (the other 9 were zeroed by the SHARPE_FLOOR=0 cutoff), so no
    renormalization is needed; the assertion below guards that invariant in
    case the upstream CSV ever changes.
    """
    w = pd.read_csv(IS_WEIGHTS_PATH, encoding="utf-8-sig")
    w.columns = w.columns.str.strip()
    w = w[w["position_type"] == POSITION_TYPE].set_index("factor")

    selected_in_is = sorted(w.index[w["selected"].astype(bool)])
    if selected_in_is != sorted(SELECTED_FACTORS):
        raise ValueError(
            f"SELECTED_FACTORS does not match the in-sample selection for "
            f"{POSITION_TYPE}.\n  in-sample: {selected_in_is}\n  here:      "
            f"{sorted(SELECTED_FACTORS)}"
        )

    weights = w.loc[SELECTED_FACTORS, "weight"].astype(float)
    total = weights.sum()
    if abs(total - 1.0) > 1e-3:
        raise ValueError(f"Frozen weights sum to {total:.6f}, expected 1.0")
    return weights / total


# ======================================================
# Windowed backtest
# ======================================================

def metrics_from_net(net: pd.Series, turnover: pd.Series) -> dict:
    """Standard metric block, using the same definitions as the in-sample run.

    Kept formula-identical to ``position_mapping.backtest_position`` so that
    in-sample and out-of-sample numbers are directly comparable: annualized
    return is the arithmetic mean x 252 (not the geometric CAGR), NAV is
    compounded, and the win rate ignores exactly-zero days.
    """
    valid = net.dropna()
    if len(valid) == 0:
        return {k: np.nan for k in (
            "ann_return", "ann_turnover", "ann_vol", "Sharpe",
            "max_drawdown", "Calmar", "win_rate", "n_days",
            "total_return", "start", "end",
        )}

    nav = (1.0 + valid).cumprod()
    ann_return = valid.mean() * TRADING_DAYS_PER_YEAR
    ann_vol = valid.std() * sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_return / ann_vol if ann_vol > EPSILON else np.nan
    max_dd = ((nav.cummax() - nav) / nav.cummax()).max()
    calmar = ann_return / max_dd if max_dd > EPSILON else np.nan
    non_zero = valid[valid != 0]
    win_rate = float((non_zero > 0).mean()) if len(non_zero) else np.nan
    ann_turnover = turnover.reindex(valid.index).mean() * TRADING_DAYS_PER_YEAR

    return {
        "ann_return":   round(float(ann_return), 4),
        "ann_turnover": round(float(ann_turnover), 4),
        "ann_vol":      round(float(ann_vol), 4),
        "Sharpe":       round(float(sharpe), 3) if not np.isnan(sharpe) else np.nan,
        "max_drawdown": round(float(max_dd), 4),
        "Calmar":       round(float(calmar), 3) if not np.isnan(calmar) else np.nan,
        "win_rate":     round(float(win_rate), 4) if not np.isnan(win_rate) else np.nan,
        "n_days":       int(len(valid)),
        "total_return": round(float(nav.iloc[-1] - 1.0), 4),
        "start":        str(valid.index.min().date()),
        "end":          str(valid.index.max().date()),
    }


def execute(position: pd.Series, simple_return: pd.Series) -> pd.DataFrame:
    """Turn a full-history position series into full-history execution results.

    T+1 lag, turnover and cost are all computed on the full history BEFORE any
    windowing, so slicing the output to the OOS window keeps the correct
    opening position at the window boundary (the first OOS day earns the
    return on the position held at the last in-sample close, and pays the cost
    of whatever rebalance happened across the boundary).
    """
    idx = position.dropna().index.intersection(simple_return.dropna().index)
    pos = position.reindex(idx)
    ret = simple_return.reindex(idx)

    pos_lag = pos.shift(1)
    turnover = pos_lag.diff().abs().fillna(0.0)
    gross = pos_lag * ret
    net = gross - turnover * COST_RATE

    return pd.DataFrame({
        "position": pos,
        "pos_executed": pos_lag,
        "asset_return": ret,
        "turnover": turnover,
        "gross_return": gross,
        "net_return": net,
    })


def window(df: pd.DataFrame, start: str | None, end: str) -> pd.DataFrame:
    mask = df.index <= end
    if start is not None:
        mask &= df.index >= start
    return df.loc[mask]


def sharpe_standard_error(net: pd.Series, sharpe_ann: float) -> float:
    """Lo (2002) standard error of an annualized Sharpe ratio under iid returns.

    SE(SR_daily) = sqrt((1 + 0.5 * SR_daily^2) / T), then annualized by
    sqrt(252). With only 462 OOS days this is the number that decides whether
    an OOS Sharpe is a result or a coin flip, so it is reported alongside
    every OOS Sharpe rather than buried in a footnote.
    """
    t = len(net.dropna())
    if t < 2 or np.isnan(sharpe_ann):
        return float("nan")
    sr_d = sharpe_ann / sqrt(TRADING_DAYS_PER_YEAR)
    return float(sqrt((1.0 + 0.5 * sr_d ** 2) / t) * sqrt(TRADING_DAYS_PER_YEAR))


# ======================================================
# Main
# ======================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("OOS TEST - 6 frozen factors, frozen direction, frozen weights")
    print(f"  evaluation window : {OOS_START} -> {OOS_END}")
    print(f"  position mapping  : {POSITION_TYPE}")
    print("=" * 74)

    # --- Step 1: frozen parameters -----------------------------------------
    direction = load_frozen_direction()
    weights = load_frozen_weights()
    print("\nStep 1: frozen in-sample parameters")
    frozen = pd.DataFrame({"direction": direction, "weight": weights})
    print(frozen.to_string())

    # --- Step 2: factors on the full history --------------------------------
    print("\nStep 2: recompute the 6 factors on full history")
    inds = load_all_indicators(INDICATOR_ROOT, PMF_TO_FILENAME)
    factor_df = pd.concat(
        [TOP15_FACTORS[name](inds) for name in SELECTED_FACTORS], axis=1,
    ).sort_index()
    factor_df = factor_df[factor_df.index <= OOS_END]
    factor_df.to_csv(OUTPUT_DIR / "oos_factor_values_6.csv")
    print(f"  factor table: {factor_df.shape[0]} rows x {factor_df.shape[1]} cols "
          f"({factor_df.index.min().date()} -> {factor_df.index.max().date()})")
    for name in SELECTED_FACTORS:
        oos_slice = factor_df.loc[factor_df.index >= OOS_START, name]
        print(f"    {name}  OOS non-null={oos_slice.notna().sum():4d}  "
              f"exact-zero share={float((oos_slice.dropna() == 0).mean()):.3f}")

    # --- Step 3: traded asset and benchmark ---------------------------------
    print("\nStep 3: load Au9995 and CSI 300")
    au = load_au9995(AU9995_PATH)
    au = au[au.index <= OOS_END].dropna()
    simple_ret = au.pct_change().rename("ret")
    print(f"  Au9995 : {len(au)} closes, {au.index.min().date()} -> {au.index.max().date()}")

    hs300 = load_hs300(HS300_PATH)
    hs300 = hs300[hs300.index <= OOS_END].dropna()
    print(f"  CSI 300: {len(hs300)} closes, {hs300.index.min().date()} -> "
          f"{hs300.index.max().date()}")

    # --- Step 4: per-factor OOS backtest ------------------------------------
    print("\nStep 4: single-factor OOS backtest (frozen direction)")
    single_rows = []
    directional_positions: dict[str, pd.Series] = {}

    for name in SELECTED_FACTORS:
        pos_mapped = position_rolling_percentage(factor_df[name])
        pos_dir = pos_mapped * int(direction[name])
        directional_positions[name] = pos_dir

        exec_full = execute(postprocess_position(pos_dir), simple_ret)
        for label, (w_start, w_end) in PERIODS.items():
            seg = window(exec_full, w_start, w_end)
            m = metrics_from_net(seg["net_return"], seg["turnover"])
            single_rows.append({
                "factor": name, "period": label,
                "position_type": POSITION_TYPE,
                "direction": int(direction[name]),
                "frozen_weight": round(float(weights[name]), 4),
                **m,
            })
        oos_m = single_rows[-1]
        is_m = single_rows[-3]
        print(f"  {name:<12s} dir={int(direction[name]):+d}  "
              f"IS Sharpe={is_m['Sharpe']:>6}  ->  OOS Sharpe={oos_m['Sharpe']:>6}  "
              f"OOS ann={oos_m['ann_return']:>8.2%}  MDD={oos_m['max_drawdown']:>6.2%}")

    single_df = pd.DataFrame(single_rows)
    single_df.to_csv(OUTPUT_DIR / "oos_single_factor_metrics.csv",
                     index=False, encoding="utf-8-sig")

    # --- Step 5: portfolio OOS backtest -------------------------------------
    # Combine the DIRECTIONAL RAW positions and post-process the sum ONCE,
    # exactly as the in-sample engine does. Post-processing each factor first
    # and then the sum would double-smooth and is not the tested configuration.
    print("\nStep 5: portfolio OOS backtest (frozen Sharpe weights)")
    pos_matrix = pd.DataFrame(directional_positions)
    pos_portfolio = postprocess_position(
        pos_matrix.mul(weights, axis=1).sum(axis=1, min_count=1)
    )
    exec_port = execute(pos_portfolio, simple_ret)

    port_rows = []
    for label, (w_start, w_end) in PERIODS.items():
        seg = window(exec_port, w_start, w_end)
        m = metrics_from_net(seg["net_return"], seg["turnover"])
        m["Sharpe_SE"] = round(sharpe_standard_error(seg["net_return"], m["Sharpe"]), 3)
        port_rows.append({
            "series": "strategy_sharpe_weighted_6factor", "period": label, **m,
        })
    strat_oos = port_rows[-1]

    oos_port = window(exec_port, OOS_START, OOS_END)
    oos_net = oos_port["net_return"].dropna()
    oos_ann_vol = float(oos_net.std() * sqrt(TRADING_DAYS_PER_YEAR))
    print(f"  strategy OOS: Sharpe={strat_oos['Sharpe']} (SE={strat_oos['Sharpe_SE']})  "
          f"ann={strat_oos['ann_return']:.2%}  vol={oos_ann_vol:.2%}  "
          f"MDD={strat_oos['max_drawdown']:.2%}  n={strat_oos['n_days']}")

    # --- Step 6: benchmarks over the same window ----------------------------
    # Two references, both over the identical 462-day window:
    #   1. Au9995 buy-and-hold  - the asset the strategy actually trades.
    #   2. CSI 300, raw and scaled to the strategy's OOS realized vol, so the
    #      absolute-return and drawdown comparison is apples-to-apples.
    zero_turnover = pd.Series(0.0, index=oos_net.index)

    def bench_row(name: str, r: pd.Series) -> dict:
        m = metrics_from_net(r, pd.Series(0.0, index=r.index))
        m["Sharpe_SE"] = round(sharpe_standard_error(r, m["Sharpe"]), 3)
        return {"series": name, "period": "out_of_sample", **m}

    au_ret_oos = window(pd.DataFrame({"r": au.pct_change()}), OOS_START, OOS_END)["r"]
    port_rows.append(bench_row(
        "Au9995_buy_and_hold", au_ret_oos.reindex(oos_net.index).dropna()))

    hs_ret_oos = window(pd.DataFrame({"r": hs300.pct_change()}), OOS_START, OOS_END)["r"].dropna()
    port_rows.append(bench_row("CSI300_buy_and_hold_raw", hs_ret_oos))

    hs_raw_vol = float(hs_ret_oos.std() * sqrt(TRADING_DAYS_PER_YEAR))
    vol_scale = oos_ann_vol / hs_raw_vol if hs_raw_vol > EPSILON else np.nan
    hs_scaled = hs_ret_oos * vol_scale
    port_rows.append(bench_row(
        f"CSI300_vol_normalized_to_{oos_ann_vol:.2%}", hs_scaled))
    # Au9995 scaled to the same vol budget - the comparison that actually
    # decides whether the signal earned its keep in 2024-2025.
    au_r = au_ret_oos.reindex(oos_net.index).dropna()
    au_scale = oos_ann_vol / float(au_r.std() * sqrt(TRADING_DAYS_PER_YEAR))
    port_rows.append(bench_row(
        f"Au9995_vol_normalized_to_{oos_ann_vol:.2%}", au_r * au_scale))
    print(f"  CSI 300 OOS raw vol={hs_raw_vol:.2%} -> scale={vol_scale:.3f} "
          f"-> target {oos_ann_vol:.2%}")

    port_df = pd.DataFrame(port_rows)
    port_df.to_csv(OUTPUT_DIR / "oos_portfolio_metrics.csv",
                   index=False, encoding="utf-8-sig")
    print()
    print(port_df[["series", "period", "n_days", "ann_return", "ann_vol",
                   "Sharpe", "Sharpe_SE", "max_drawdown", "Calmar", "win_rate",
                   "total_return"]].to_string(index=False))

    # --- Step 7: daily and monthly output -----------------------------------
    nav_daily = oos_port.copy()
    nav_daily["nav"] = (1.0 + nav_daily["net_return"].fillna(0.0)).cumprod()
    nav_daily["nav_au9995_bh"] = (
        1.0 + au_ret_oos.reindex(nav_daily.index).fillna(0.0)
    ).cumprod()
    nav_daily["nav_csi300_volnorm"] = (
        1.0 + hs_scaled.reindex(nav_daily.index).fillna(0.0)
    ).cumprod()
    nav_daily.index.name = "date"
    nav_daily.to_csv(OUTPUT_DIR / "oos_nav_daily.csv", encoding="utf-8-sig")

    monthly = oos_net.groupby(pd.Grouper(freq="ME")).apply(
        lambda x: (1.0 + x).prod() - 1.0
    ).rename("net_return")
    monthly.index = monthly.index.strftime("%Y-%m")
    monthly.index.name = "month"
    monthly.round(4).to_csv(OUTPUT_DIR / "oos_monthly_returns.csv",
                            encoding="utf-8-sig")
    print(f"\n  monthly returns ({len(monthly)} months, "
          f"{int((monthly > 0).sum())} positive)")

    # --- Step 8: return attribution -----------------------------------------
    # The headline OOS Sharpe means nothing until we know whether the signal
    # earned it or whether it is passive exposure to a gold market that rose
    # ~97% over the window. Regress the strategy's daily net return on the
    # Au9995 daily return: net_t = alpha + beta * r_t + e_t.
    print("\nStep 8: return attribution vs Au9995")
    pos_exec = oos_port["pos_executed"].reindex(oos_net.index)
    r_asset = oos_port["asset_return"].reindex(oos_net.index)

    beta = float(np.cov(oos_net, r_asset)[0, 1] / np.var(r_asset, ddof=1))
    alpha_daily = float(oos_net.mean() - beta * r_asset.mean())
    resid = oos_net - beta * r_asset
    resid_vol = float(resid.std() * sqrt(TRADING_DAYS_PER_YEAR))
    alpha_t = alpha_daily / (resid.std() / sqrt(len(oos_net)))
    corr = float(oos_net.corr(r_asset))

    attribution = {
        "mean_executed_position":   round(float(pos_exec.mean()), 4),
        "share_days_long":          round(float((pos_exec > 0).mean()), 4),
        "share_days_short":         round(float((pos_exec < 0).mean()), 4),
        "mean_abs_position":        round(float(pos_exec.abs().mean()), 4),
        "beta_to_Au9995":           round(beta, 4),
        "corr_to_Au9995":           round(corr, 4),
        "r_squared_to_Au9995":      round(corr ** 2, 4),
        "ann_return_total":         round(float(oos_net.mean() * TRADING_DAYS_PER_YEAR), 4),
        "ann_return_from_beta":     round(float(beta * r_asset.mean() * TRADING_DAYS_PER_YEAR), 4),
        "ann_return_alpha":         round(float(alpha_daily * TRADING_DAYS_PER_YEAR), 4),
        "alpha_t_stat":             round(float(alpha_t), 3),
        "residual_ann_vol":         round(resid_vol, 4),
        "Sharpe":                   strat_oos["Sharpe"],
        "Sharpe_SE":                strat_oos["Sharpe_SE"],
        "Sharpe_95CI_low":          round(strat_oos["Sharpe"] - 1.96 * strat_oos["Sharpe_SE"], 3),
        "Sharpe_95CI_high":         round(strat_oos["Sharpe"] + 1.96 * strat_oos["Sharpe_SE"], 3),
    }
    for year, grp in oos_net.groupby(oos_net.index.year):
        attribution[f"Sharpe_{year}"] = round(
            float(grp.mean() / grp.std() * sqrt(TRADING_DAYS_PER_YEAR)), 3)
        attribution[f"ann_return_{year}"] = round(
            float(grp.mean() * TRADING_DAYS_PER_YEAR), 4)

    pd.Series(attribution, name="value").to_frame().to_csv(
        OUTPUT_DIR / "oos_attribution.csv", encoding="utf-8-sig",
        index_label="metric")
    for k, v in attribution.items():
        print(f"    {k:<26s} {v}")

    # --- Step 9: charts ------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"hspace": 0.32})

    ax = axes[0]
    ax.plot(nav_daily.index, nav_daily["nav"], color="steelblue", lw=1.6,
            label=f"Strategy (6 factors, frozen)  Sharpe={strat_oos['Sharpe']}")
    ax.plot(nav_daily.index, nav_daily["nav_csi300_volnorm"], color="gray",
            lw=1.1, ls="--",
            label=f"CSI 300 vol-normalized to {oos_ann_vol:.1%}")
    ax.plot(nav_daily.index, nav_daily["nav_au9995_bh"], color="goldenrod",
            lw=1.1, ls=":", label="Au9995 buy-and-hold (unscaled)")
    ax.axhline(1.0, color="black", lw=0.5, ls=":", alpha=0.6)
    ax.set_title(
        f"Out-of-sample NAV  {oos_net.index.min().date()} -> "
        f"{oos_net.index.max().date()}  ({len(oos_net)} trading days)\n"
        f"ann={strat_oos['ann_return']:.2%}  vol={oos_ann_vol:.2%}  "
        f"Sharpe={strat_oos['Sharpe']} (SE={strat_oos['Sharpe_SE']})  "
        f"MDD={strat_oos['max_drawdown']:.2%}",
        fontsize=10, fontweight="bold")
    ax.set_ylabel("NAV")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, name in enumerate(SELECTED_FACTORS):
        e = window(execute(postprocess_position(directional_positions[name]),
                           simple_ret), OOS_START, OOS_END)
        nav_f = (1.0 + e["net_return"].fillna(0.0)).cumprod()
        ax.plot(nav_f.index, nav_f.values, lw=1.0, color=colors[i], alpha=0.85,
                label=f"{name} (w={weights[name]:.3f})")
    ax.axhline(1.0, color="black", lw=0.5, ls=":", alpha=0.6)
    ax.set_title("Per-factor OOS NAV (frozen direction, same window)", fontsize=10)
    ax.set_ylabel("NAV")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.savefig(PLOTS_DIR / "oos_portfolio_nav.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Continuous IS -> OOS NAV, so the regime break is visible at a glance.
    cont = window(exec_port, IS_FULL_START, OOS_END)
    cont_nav = (1.0 + cont["net_return"].fillna(0.0)).cumprod()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(cont_nav.index, cont_nav.values, color="steelblue", lw=1.2)
    split = pd.Timestamp(OOS_START)
    ax.axvline(split, color="crimson", lw=1.2, ls="--")
    ax.annotate("in-sample selection ends", xy=(split, cont_nav.max()),
                xytext=(-8, -4), textcoords="offset points",
                ha="right", va="top", fontsize=9, color="crimson")
    ax.axhline(1.0, color="black", lw=0.5, ls=":", alpha=0.6)
    is_row = next(r for r in port_rows if r["period"] == "in_sample_full")
    ax.set_title(
        f"Continuous NAV, same frozen configuration\n"
        f"in-sample {is_row['start']} -> {is_row['end']}: Sharpe={is_row['Sharpe']}  |  "
        f"out-of-sample {strat_oos['start']} -> {strat_oos['end']}: "
        f"Sharpe={strat_oos['Sharpe']} (SE={strat_oos['Sharpe_SE']})",
        fontsize=10, fontweight="bold")
    ax.set_ylabel("NAV")
    ax.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "oos_is_vs_oos_nav.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 74)
    print(f"Done. Outputs in: {OUTPUT_DIR}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
