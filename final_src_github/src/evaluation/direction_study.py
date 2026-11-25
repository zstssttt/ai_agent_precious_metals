"""Direction-setting study — how is each factor's trade sign actually chosen?

Why this study exists
---------------------
``src/backtest/position_mapping.py`` fixes each factor's trade direction from a
**full-sample Spearman IC against the contemporaneous daily return**::

    simple_ret = au_price.pct_change()              # r_t = (P_t - P_{t-1}) / P_{t-1}
    ic_df      = compute_ic_direction(factor_df, simple_ret)
    direction  = int(np.sign(ic))

Two problems with that, both of which this module quantifies rather than asserts:

1. **Wrong alignment.** ``pct_change()`` is the return *of the same day* as the
   factor value, so the IC measures co-movement, not prediction. The backtest
   however trades ``pos.shift(1)``, i.e. ``factor_{t-1} -> r_t``. The quantity
   used to pick the sign is not the quantity that earns the money.
2. **No significance.** The 15 contemporaneous ICs range over |IC| = 0.003 to
   0.083. Because the factors are heavily smoothed (1st-order autocorrelation
   0.67-0.91), the effective sample size is 229-925, not 4,615. Every single
   |t| comes out below 1.1 - the signs are statistically indistinguishable from
   coin flips.

So the sign of every factor - and therefore the sign of the whole portfolio -
rests on a step that carries no measurable information. This module runs the
same engine four ways to measure how much that step actually matters.

The four methods
----------------
=========================  ====================================================
``contemporaneous_full``   Status quo. ``sign(spearman(f_t, r_t))``, full sample.
``forward_full``           ``sign(spearman(f_t, lnP_{t+1} - lnP_t))``, full
                           sample. Same statistic, but aligned to what the
                           backtest actually trades. Still in-sample.
``economic_prior``         Sign from the factor's economic hypothesis, fitted on
                           zero data. A factor whose construction admits no
                           a-priori sign (see ``NO_PRIOR`` below) is dropped
                           rather than guessed.
``walkforward_forward``    ``sign`` of the *expanding-window* forward IC,
                           recomputed every ``DIRECTION_REFIT_DAYS`` using only
                           data available at that point. Time-varying, and the
                           only method of the four with no look-ahead at all.
=========================  ====================================================

Why nine factors get an economic prior and six do not
-----------------------------------------------------
Nine of the fifteen factors are built as ``direction x intensity`` where
``intensity >= 0``, so ``sign(factor) == direction`` and ``direction`` is a
named economic state ("the dollar is trending up", "GPR is escalating"). The
sign of the gold response to that state is a textbook claim, not a fitted
parameter.

The other six are *changes in a rolling correlation between two indicators*
(``corr_20d - corr_20d.shift(20)``, optionally divided by a risk proxy). There
is no economic theory that says "the DXY/ETF-flow correlation strengthening
implies gold goes up" rather than down. Their sign is only obtainable by
fitting, which makes them the most fragile factors in the book - and they carry
44.2% of the Sharpe-weighted portfolio. Reporting that number is half the point
of this study.

Protocol
--------
* Factors are rebuilt over the full available history so the rolling percentile
  mapping and all ``ts_*`` operators are warm before the evaluation windows.
* Only ``rolling_percentage`` is used (the mapping selected in-sample), so the
  mapping choice cannot confound the direction comparison.
* **Equal weighting is the headline portfolio.** Sharpe-proportional weights are
  themselves fitted in-sample, which would mix a second fitting step into a
  study about the first one. Sharpe-weighted results are reported alongside,
  with weights fitted on the in-sample window only and frozen for OOS, matching
  the protocol in ``OOS/run_oos.py``.
* Position post-processing, T+1 lag and the 3 bp linear cost are imported from
  ``position_mapping`` unchanged, so every number here is comparable to the
  numbers already in ``results/``.

Outputs (under ``results/direction_study/``)
--------------------------------------------
    factor_values_full.csv      15 factors over the full history (cached)
    direction_table.csv         per factor x method: sign, IC, t-stat, n_eff
    single_factor_metrics.csv   15 factors x 4 methods x 3 windows
    portfolio_metrics.csv       EW and SW portfolios x 4 methods x 3 windows
    prior_vs_data.csv           does the data agree with the economic prior?
    plots/direction_nav.png     portfolio NAV per method, IS/OOS split marked

Usage
-----
    export GOLD_FACTOR_ALIGNED_ROOT="/path/to/补齐后数据/补齐后数据"
    export AU9995_PATH="$GOLD_FACTOR_ALIGNED_ROOT/补齐后日频数据2.0/收盘价_Au9995_S0035818.csv"
    python -m src.evaluation.direction_study            # full study
    python -m src.evaluation.direction_study --is-only  # skip the OOS window
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

matplotlib.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest import position_mapping as pm  # noqa: E402
from src.factors import build_factors as bf  # noqa: E402

logger = logging.getLogger(__name__)


# ======================================================
# Configuration
# ======================================================

OUTPUT_DIR = Path(os.environ.get(
    "DIRECTION_STUDY_OUTPUT_DIR", REPO_ROOT / "results" / "direction_study"
))

# The mapping selected in-sample. Held fixed so it cannot confound the study.
POSITION_TYPE = "rolling_percentage"

# Evaluation windows. IS_END matches position_mapping.CUTOFF_DATE; OOS_END is the
# last genuine Au9995 print (see OOS/run_oos.py for why it is not later).
IS_END = "2023-12-31"
IS_2015_START = "2015-01-01"
OOS_START = "2024-01-01"
OOS_END = "2025-11-28"

# Walk-forward direction: refit cadence and minimum history before any position.
DIRECTION_REFIT_DAYS = 21
DIRECTION_MIN_OBS = 504          # ~2 years of forward-IC observations

# A fitted sign only overrides an economic prior when the data rejects the prior
# at this |t|. With autocorrelation-adjusted standard errors no factor in this
# book clears it, which is itself a reportable result.
PRIOR_REJECT_T = 2.0


# ======================================================
# Economic priors
# ======================================================
# For each factor: (prior_sign, economic claim).
#
# prior_sign = +1  ->  a positive factor value predicts gold UP
# prior_sign = -1  ->  a positive factor value predicts gold DOWN
# prior_sign =  0  ->  the construction admits no a-priori sign; factor dropped
#
# The nine signed entries are all `direction x intensity` factors with
# intensity >= 0, so sign(factor) == direction and the claim below is exactly
# the claim the factor makes.

ECONOMIC_PRIOR: dict[str, tuple[int, str]] = {
    # --- Theme 1: Dollar -------------------------------------------------
    "pmf_069545": (-1, "direction = sign(DXY 20d MA - 60d MA) = dollar strengthening. "
                       "Gold is priced in USD, so a stronger dollar is a lower gold price."),
    # --- Theme 2: Gold price / basis -------------------------------------
    "pmf_051113": (+1, "direction = sign(GPR 5d MA trend) = geopolitical risk escalating. "
                       "Safe-haven demand bids gold up."),
    # --- Theme 3: ETF / positioning --------------------------------------
    "pmf_104203": (+1, "signal = Aroon balance on SPDR holdings returns = inflow momentum. "
                       "ETF creations are net new investment demand for bullion."),
    "pmf_058341": (+1, "direction = sign(SPDR holdings 5d MA trend) = ETF inflow. "
                       "Same claim as pmf_104203 on a trend rather than a breakout."),
    # --- Theme 4: Inflation / term structure -----------------------------
    "pmf_102135": (+1, "signal = sign(CPI 5d MA - 20d MA) = inflation accelerating. "
                       "Gold is held as an inflation hedge. NOTE: Erb & Harvey (2013) show "
                       "this link is weak at horizons under a decade - the weakest of the nine."),
    "pmf_118259": (+1, "direction = sign(CPI 20d MA - 60d MA) = medium-term inflation rising. "
                       "Same claim as pmf_102135 on a slower window. The Bollinger `position` "
                       "term can be negative, so sign(factor) != direction everywhere; the "
                       "prior applies to the direction component."),
    # --- Theme 5: Risk / volatility --------------------------------------
    "pmf_051056": (+1, "direction = sign(GPR 5d MA trend) = geopolitical risk escalating. "
                       "Same claim as pmf_051113 with a different intensity term."),
    "pmf_055491": (+1, "direction = sign(gold ATM IV 20d MA trend) = implied vol rising = "
                       "the market is pricing more fear. Gold is a crisis hedge."),
    "pmf_066409": (+1, "direction = sign(GPR 20d MA - 60d MA) = medium-term geopolitical "
                       "risk rising. Slower-window version of pmf_051056."),

    # --- No derivable prior ----------------------------------------------
    # All six are `change in a rolling correlation between two indicators`,
    # optionally divided by a risk proxy. "The DXY / ETF-flow correlation got
    # stronger" has no signed implication for the gold price in any model.
    "pmf_128453": (0, "Delta corr(DXY diff, SPDR flow diff) / GPR - a correlation change. "
                      "No economic theory signs it."),
    "pmf_011095": (0, "OLS slope of corr(DXY diff, SPDR flow return) - a correlation trend. "
                      "No economic theory signs it."),
    "pmf_127111": (0, "Delta corr(Au9995 diff, SPDR diff) / implied vol. "
                      "No economic theory signs it."),
    "pmf_125861": (0, "Delta corr(COMEX diff, SPDR cash diff) / GPR. "
                      "No economic theory signs it."),
    "pmf_131810": (0, "Delta corr(ETF holdings diff, lagged 10Y yield return) / GPR. "
                      "No economic theory signs it."),
    "pmf_003678": (0, "Delta corr(TIPS diff, lagged ETF holdings diff). "
                      "No economic theory signs it."),
}

NO_PRIOR = [f for f, (s, _) in ECONOMIC_PRIOR.items() if s == 0]


# ======================================================
# Section 1: data
# ======================================================

def load_factors_full_history(cache_path: Path, rebuild: bool) -> pd.DataFrame:
    """Build (or load from cache) the 15 factors over the full available history.

    ``build_factors.build_factors`` truncates nothing; the CUTOFF_DATE in that
    module is applied only in its ``main()``. Computing on full history keeps the
    rolling operators warm so the OOS window does not start on a cold cache.
    """
    if cache_path.exists() and not rebuild:
        logger.info("Loading cached factor table: %s", cache_path)
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.sort_index()

    logger.info("Rebuilding factors from %s", bf.INDICATOR_ROOT)
    df = bf.build_factors(bf.INDICATOR_ROOT, bf.PMF_TO_FILENAME)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    logger.info("Cached %d x %d factor table to %s", *df.shape, cache_path)
    return df


def load_prices() -> tuple[pd.Series, pd.Series]:
    """Return (au9995_close, simple_daily_return) over the full history."""
    price = pm.load_au9995(pm.AU9995_PATH)
    price = price[price.index <= OOS_END]
    return price, price.pct_change().rename("ret")


# ======================================================
# Section 2: the four direction estimators
# ======================================================

def _effective_n(series: pd.Series, n: int) -> float:
    """Autocorrelation-adjusted sample size: n_eff = n * (1 - rho) / (1 + rho).

    The factors are smoothed by construction (ts_mean / ts_delay), so daily
    observations are far from independent. Using the raw n inflates every
    t-statistic by sqrt(n / n_eff) - a factor of 2 to 4.5 for this book.
    """
    rho = series.dropna().autocorr(1)
    if rho is None or np.isnan(rho) or rho >= 0.999:
        return float("nan")
    return float(n) * (1.0 - rho) / (1.0 + rho)


def _spearman_with_t(
    factor: pd.Series, target: pd.Series
) -> tuple[float, int, float, float]:
    """Spearman IC plus an autocorrelation-adjusted t-stat.

    Returns (ic, n_obs, n_eff, t_eff). ``t_eff = ic * sqrt(n_eff)`` is the usual
    small-rho approximation; it is deliberately the conservative choice.
    """
    pair = pd.concat([factor, target], axis=1).dropna()
    if len(pair) < 30:
        return float("nan"), len(pair), float("nan"), float("nan")
    ic, _ = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    n = len(pair)
    n_eff = _effective_n(pair.iloc[:, 0], n)
    t_eff = ic * np.sqrt(n_eff) if not np.isnan(n_eff) else float("nan")
    return float(ic), n, n_eff, float(t_eff)


def direction_contemporaneous(
    factor: pd.Series, contemp_ret: pd.Series, **_: object
) -> tuple[pd.Series | int, dict]:
    """Status quo: sign of the full-sample IC against the SAME-day return."""
    ic, n, n_eff, t = _spearman_with_t(factor, contemp_ret)
    sign = int(np.sign(ic)) if not np.isnan(ic) else 0
    return sign, {"ic": ic, "n_obs": n, "n_eff": n_eff, "t_eff": t, "basis": "contemporaneous r_t"}


def direction_forward(
    factor: pd.Series, fwd_ret: pd.Series, **_: object
) -> tuple[pd.Series | int, dict]:
    """Same statistic, aligned to what the backtest trades (next-day return)."""
    ic, n, n_eff, t = _spearman_with_t(factor, fwd_ret)
    sign = int(np.sign(ic)) if not np.isnan(ic) else 0
    return sign, {"ic": ic, "n_obs": n, "n_eff": n_eff, "t_eff": t, "basis": "forward r_{t+1}"}


def direction_economic_prior(
    factor: pd.Series, fwd_ret: pd.Series, name: str = "", **_: object
) -> tuple[pd.Series | int, dict]:
    """Sign from the economic hypothesis; fitted on zero data.

    The forward IC is still computed, but only to check whether the data
    *rejects* the prior at |t| > PRIOR_REJECT_T. If it does, the factor is
    dropped (sign 0) and logged as "hypothesis rejected" - which is a research
    finding, not a parameter.
    """
    prior, claim = ECONOMIC_PRIOR.get(name, (0, "unmapped factor"))
    ic, n, n_eff, t = _spearman_with_t(factor, fwd_ret)

    sign, note = prior, "prior applied"
    if prior == 0:
        note = "no derivable prior -> dropped"
    elif not np.isnan(t) and abs(t) > PRIOR_REJECT_T and np.sign(ic) != prior:
        sign, note = 0, f"prior REJECTED by data (t={t:.2f}) -> dropped"

    return sign, {"ic": ic, "n_obs": n, "n_eff": n_eff, "t_eff": t,
                  "basis": "economic prior", "prior": prior, "note": note,
                  "claim": claim}


def direction_walkforward(
    factor: pd.Series, fwd_ret: pd.Series, **_: object
) -> tuple[pd.Series | int, dict]:
    """Expanding-window forward IC, refit every DIRECTION_REFIT_DAYS.

    At each refit date the sign uses only observations strictly up to that date,
    so this is the only estimator here with no look-ahead. Before
    DIRECTION_MIN_OBS pairs are available the direction is 0 (flat), which costs
    real return but is the honest treatment of "I did not know yet".
    """
    pair = pd.concat([factor, fwd_ret], axis=1).dropna()
    pair.columns = ["f", "y"]
    if len(pair) < DIRECTION_MIN_OBS:
        return 0, {"ic": float("nan"), "n_obs": len(pair), "n_eff": float("nan"),
                   "t_eff": float("nan"), "basis": "walk-forward (insufficient history)"}

    signs = pd.Series(index=pair.index, dtype=float)
    flips = 0
    last = 0
    for i in range(DIRECTION_MIN_OBS, len(pair) + 1, DIRECTION_REFIT_DAYS):
        hist = pair.iloc[:i]
        ic, _ = spearmanr(hist["f"], hist["y"])
        s = int(np.sign(ic)) if not np.isnan(ic) else 0
        if last != 0 and s != last:
            flips += 1
        last = s
        end = min(i + DIRECTION_REFIT_DAYS, len(pair))
        signs.iloc[i - 1:end] = s

    signs = signs.reindex(factor.index).ffill().fillna(0.0)
    ic_final, n, n_eff, t = _spearman_with_t(factor, fwd_ret)
    return signs, {"ic": ic_final, "n_obs": n, "n_eff": n_eff, "t_eff": t,
                   "basis": f"walk-forward, refit/{DIRECTION_REFIT_DAYS}d",
                   "note": f"{flips} sign flips, final sign {int(last):+d}"}


METHODS: dict[str, Callable] = {
    "contemporaneous_full": direction_contemporaneous,
    "forward_full": direction_forward,
    "economic_prior": direction_economic_prior,
    "walkforward_forward": direction_walkforward,
}


# ======================================================
# Section 3: execution helpers (reused from the OOS runner verbatim)
# ======================================================
# Importing rather than re-implementing guarantees that every number this study
# prints is computed by the same code that produced results/ and OOS/results/,
# so the status-quo row below must reproduce the published Sharpe of 0.939.

def _load_oos_helpers():
    """Import OOS/run_oos.py by path (it is a script, not an importable package)."""
    import importlib.util

    path = REPO_ROOT / "OOS" / "run_oos.py"
    spec = importlib.util.spec_from_file_location("_run_oos", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ======================================================
# Section 4: one method, end to end
# ======================================================

def run_method(
    method_name: str,
    factors: pd.DataFrame,
    contemp_ret: pd.Series,
    fwd_ret: pd.Series,
    simple_ret: pd.Series,
    oos: object,
    periods: dict[str, tuple[Optional[str], str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """Run all 15 factors under one direction-setting method.

    Returns (direction_rows, single_factor_rows, directional_positions).
    ``directional_positions`` holds the pre-post-processing signed positions, so
    the portfolio can combine them and post-process the sum once - matching the
    convention in ``position_mapping.main()``.
    """
    estimator = METHODS[method_name]
    map_fn = pm.POSITION_FUNCS[POSITION_TYPE]

    dir_rows, single_rows = [], []
    directional: dict[str, pd.Series] = {}

    for name in factors.columns:
        raw = factors[name]
        target = contemp_ret if method_name == "contemporaneous_full" else fwd_ret
        direction, diag = estimator(raw, target, name=name)

        is_series = isinstance(direction, pd.Series)
        dir_rows.append({
            "method": method_name,
            "factor": name,
            "direction": ("time-varying" if is_series
                          else f"{int(direction):+d}" if direction else "0 (dropped)"),
            "ic": round(diag["ic"], 5) if not np.isnan(diag.get("ic", np.nan)) else np.nan,
            "n_obs": diag.get("n_obs"),
            "n_eff": (round(diag["n_eff"]) if not np.isnan(diag.get("n_eff", np.nan))
                      else np.nan),
            "t_eff": (round(diag["t_eff"], 2) if not np.isnan(diag.get("t_eff", np.nan))
                      else np.nan),
            "significant_at_|t|>2": (
                "yes" if (not np.isnan(diag.get("t_eff", np.nan))
                          and abs(diag["t_eff"]) > PRIOR_REJECT_T) else "no"),
            "basis": diag.get("basis", ""),
            "note": diag.get("note", ""),
        })

        # A dropped factor holds no position and is excluded from the portfolio.
        if not is_series and direction == 0:
            continue

        pos_dir = map_fn(raw) * direction
        directional[name] = pos_dir

        exec_df = oos.execute(pm.postprocess_position(pos_dir), simple_ret)
        for label, (start, end) in periods.items():
            w = oos.window(exec_df, start, end)
            m = oos.metrics_from_net(w["net_return"], w["turnover"])
            single_rows.append({"method": method_name, "factor": name,
                                "period": label, **m})

    return pd.DataFrame(dir_rows), pd.DataFrame(single_rows), directional


def build_portfolios(
    method_name: str,
    directional: dict[str, pd.Series],
    single_df: pd.DataFrame,
    simple_ret: pd.Series,
    oos: object,
    periods: dict[str, tuple[Optional[str], str]],
    is_label: str,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame]:
    """Combine the signed positions into an equal-weighted and a Sharpe-weighted book.

    Equal weighting is the headline: it introduces no fitted parameter, so the
    only thing that differs across methods is the direction. Sharpe-proportional
    weights are fitted on the in-sample window *only* and then frozen, matching
    ``OOS/run_oos.py``; they are reported for comparability with
    ``results/portfolio_metrics.csv``, not as the primary result.
    """
    if not directional:
        return pd.DataFrame(), {}, pd.DataFrame()

    pos_matrix = pd.DataFrame(directional)

    is_sharpes = (single_df[single_df["period"] == is_label]
                  .set_index("factor")["Sharpe"].fillna(0.0)
                  .reindex(pos_matrix.columns).fillna(0.0))
    raw_w = (is_sharpes - pm.SHARPE_FLOOR).clip(lower=0.0)
    total = raw_w.sum()
    weights = (raw_w / total if total > pm.EPSILON
               else pd.Series(0.0, index=is_sharpes.index))

    books = {
        # Equal weight over every factor that has a direction under this method.
        "equal_weighted": pos_matrix.mean(axis=1),
        # Identical to position_mapping.py:632 - including its `.sum(min_count=1)`
        # behaviour, so the status-quo row reproduces the published number.
        "sharpe_weighted": pos_matrix.mul(weights, axis=1).sum(axis=1, min_count=1),
    }

    rows, navs = [], {}
    for book, pos in books.items():
        exec_df = oos.execute(pm.postprocess_position(pos), simple_ret)
        for label, (start, end) in periods.items():
            w = oos.window(exec_df, start, end)
            m = oos.metrics_from_net(w["net_return"], w["turnover"])
            m["Sharpe_SE"] = round(
                oos.sharpe_standard_error(w["net_return"], m["Sharpe"]), 3)
            rows.append({
                "method": method_name, "portfolio": book, "period": label,
                "n_factors": int(pos_matrix.shape[1]),
                "n_weighted": int((weights > 0).sum()) if book == "sharpe_weighted"
                              else int(pos_matrix.shape[1]),
                **m,
            })
        navs[book] = (1.0 + exec_df["net_return"].fillna(0.0)).cumprod()

    weight_df = pd.DataFrame({
        "method": method_name, "factor": weights.index,
        "is_sharpe": is_sharpes.round(3).values, "weight": weights.round(4).values,
    })
    return pd.DataFrame(rows), navs, weight_df


# ======================================================
# Section 5: reporting
# ======================================================

def plot_methods(
    navs_by_method: dict[str, pd.Series], book: str, save_path: Path
) -> None:
    """One NAV curve per direction method, with the IS/OOS boundary marked."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for method, nav in navs_by_method.items():
        s = nav.dropna()
        if len(s):
            ax.plot(s.index, s.values, linewidth=1.5, label=method)

    ax.axvline(pd.Timestamp(OOS_START), color="0.35", linestyle="--", linewidth=1.2)
    ax.text(pd.Timestamp(OOS_START), ax.get_ylim()[1], "  OOS starts",
            va="top", ha="left", fontsize=9, color="0.35")
    ax.set_yscale("log")
    ax.set_title(f"Direction-setting method vs NAV — {book}, {POSITION_TYPE}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("NAV (log scale)")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


def prior_vs_data_table(factors: pd.DataFrame, fwd_ret: pd.Series) -> pd.DataFrame:
    """Does the forward IC agree with the economic prior, and is it significant?"""
    rows = []
    for name in factors.columns:
        prior, claim = ECONOMIC_PRIOR.get(name, (0, "unmapped"))
        ic, n, n_eff, t = _spearman_with_t(factors[name], fwd_ret)
        data_sign = int(np.sign(ic)) if not np.isnan(ic) else 0
        if prior == 0:
            verdict = "no prior (correlation-change factor)"
        elif data_sign == 0:
            verdict = "no data"
        elif data_sign == prior:
            verdict = "agrees" + ("" if abs(t) > PRIOR_REJECT_T else " (but |t|<2)")
        else:
            verdict = ("DISAGREES, data wins (|t|>2)" if abs(t) > PRIOR_REJECT_T
                       else "disagrees, but |t|<2 so prior kept")
        rows.append({
            "factor": name, "prior": prior if prior else 0,
            "forward_ic": round(ic, 5) if not np.isnan(ic) else np.nan,
            "data_sign": data_sign, "n_eff": (round(n_eff) if not np.isnan(n_eff)
                                              else np.nan),
            "t_eff": round(t, 2) if not np.isnan(t) else np.nan,
            "verdict": verdict, "economic_claim": claim,
        })
    return pd.DataFrame(rows).set_index("factor")


# ======================================================
# Main
# ======================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--is-only", action="store_true",
                   help="Evaluate the in-sample windows only (skip 2024-2025).")
    p.add_argument("--rebuild-factors", action="store_true",
                   help="Recompute the factor table instead of using the cache.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    oos = _load_oos_helpers()

    print("=" * 78)
    print("DIRECTION-SETTING STUDY")
    print(f"  position mapping : {POSITION_TYPE}  (held fixed)")
    print(f"  methods          : {', '.join(METHODS)}")
    print(f"  factors with an economic prior : {15 - len(NO_PRIOR)}/15"
          f"   (no prior: {len(NO_PRIOR)})")
    print("=" * 78)

    # --- Step 1: factors and returns -------------------------------------
    factors = load_factors_full_history(
        OUTPUT_DIR / "factor_values_full.csv", args.rebuild_factors)
    factors = factors[factors.index <= OOS_END]
    price, simple_ret = load_prices()
    # The backtest earns r_t on the position held from t-1, so the factor at t
    # is predicting r_{t+1}. Using exactly that series as the IC target is the
    # whole point of the `forward_*` methods.
    fwd_ret = simple_ret.shift(-1).rename("fwd_ret")
    print(f"\nFactors : {factors.shape[0]} rows x {factors.shape[1]} cols  "
          f"{factors.index.min().date()} -> {factors.index.max().date()}")
    print(f"Au9995  : {len(price)} rows  "
          f"{price.index.min().date()} -> {price.index.max().date()}")

    periods: dict[str, tuple[Optional[str], str]] = {
        "in_sample_full": (None, IS_END),
        "in_sample_2015plus": (IS_2015_START, IS_END),
    }
    if not args.is_only:
        periods["out_of_sample"] = (OOS_START, OOS_END)

    # --- Step 2: run every method ----------------------------------------
    all_dirs, all_single, all_port, all_weights = [], [], [], []
    navs_ew, navs_sw = {}, {}

    for method in METHODS:
        print(f"\n{'-' * 78}\n>>> {method}\n{'-' * 78}")
        dir_df, single_df, directional = run_method(
            method, factors, simple_ret, fwd_ret, simple_ret, oos, periods)
        all_dirs.append(dir_df)
        all_single.append(single_df)

        kept = list(directional)
        print(f"  factors carrying a position : {len(kept)}/15")
        dropped = [f for f in factors.columns if f not in kept]
        if dropped:
            print(f"  dropped                     : {', '.join(dropped)}")

        port_df, navs, weight_df = build_portfolios(
            method, directional, single_df, simple_ret, oos, periods,
            "in_sample_full")
        all_port.append(port_df)
        all_weights.append(weight_df)
        if navs:
            navs_ew[method] = navs["equal_weighted"]
            navs_sw[method] = navs["sharpe_weighted"]

        for _, r in port_df.iterrows():
            print(f"  [{r['portfolio']:<16s}] {r['period']:<19s} "
                  f"Sharpe={r['Sharpe']:>6} (SE {r['Sharpe_SE']:>5})  "
                  f"ann={r['ann_return']:>8.2%}  MDD={r['max_drawdown']:>7.2%}")

    # --- Step 3: write artifacts -----------------------------------------
    dir_table = pd.concat(all_dirs, ignore_index=True)
    single_table = pd.concat(all_single, ignore_index=True)
    port_table = pd.concat(all_port, ignore_index=True)
    weight_table = pd.concat(all_weights, ignore_index=True)
    prior_table = prior_vs_data_table(factors, fwd_ret)

    for df, fname in (
        (dir_table, "direction_table.csv"),
        (single_table, "single_factor_metrics.csv"),
        (port_table, "portfolio_metrics.csv"),
        (weight_table, "portfolio_weights.csv"),
        (prior_table.reset_index(), "prior_vs_data.csv"),
    ):
        df.to_csv(OUTPUT_DIR / fname, index=False, encoding="utf-8-sig")
    print(f"\n{'=' * 78}\nArtifacts written to {OUTPUT_DIR}")

    plot_methods(navs_ew, "equal_weighted", OUTPUT_DIR / "plots" / "direction_nav_ew.png")
    plot_methods(navs_sw, "sharpe_weighted", OUTPUT_DIR / "plots" / "direction_nav_sw.png")

    # --- Step 4: replication check ---------------------------------------
    # The status quo row must reproduce results/portfolio_metrics.csv
    # (Sharpe 0.939, ann_return 5.12%). If it does not, this study's engine has
    # drifted from the published one and nothing below should be trusted.
    check = port_table[
        (port_table["method"] == "contemporaneous_full")
        & (port_table["portfolio"] == "sharpe_weighted")
        & (port_table["period"] == "in_sample_full")
    ]
    print(f"\n{'=' * 78}\nREPLICATION CHECK (status quo vs published results/)")
    if len(check):
        got_s, got_r = float(check["Sharpe"].iloc[0]), float(check["ann_return"].iloc[0])
        ok = abs(got_s - 0.939) < 0.02 and abs(got_r - 0.0512) < 0.002
        print(f"  published : Sharpe 0.939   ann_return 5.12%")
        print(f"  this run  : Sharpe {got_s:<7.3f} ann_return {got_r:.2%}")
        print(f"  -> {'MATCH' if ok else 'MISMATCH - investigate before reading further'}")
    print("=" * 78)

    # --- Step 5: headline comparison -------------------------------------
    print("\nHEADLINE: equal-weighted portfolio (no fitted weights)\n")
    pivot = port_table[port_table["portfolio"] == "equal_weighted"].pivot(
        index="method", columns="period", values="Sharpe")
    pivot = pivot.reindex(list(METHODS))
    print(pivot.to_string())
    print("\nPrior vs data (forward IC):\n")
    print(prior_table[["prior", "forward_ic", "data_sign", "n_eff", "t_eff",
                       "verdict"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
