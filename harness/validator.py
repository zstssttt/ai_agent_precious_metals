"""④ Validation Unit — the project's truth source. No LLM (C1).

Turns a spec into robustness statistics by scanning its window grid. Everything
here is deterministic: same spec in, same numbers out, no clock, no network, no
randomness.

Three conventions are load-bearing and were each chosen against the alternative
after measuring. They are documented at the function that implements them:

1. **Direction is fitted on train only** (`fit_direction`). Upstream fits it on
   the full sample, which is why CLAUDE.md A7 forbids comparing to the
   published 0.939.
2. **Positions are mapped on the full series, then sliced** (`_positions`).
   Slicing first would burn the 252-day warm-up inside each split and cost over
   half the validation window.
3. **A cell "passes" on GATE_1/2/4 only** (`config.CELL_PASS_GATES`). GATE_3 is
   pass_rate itself, so including it would be circular.

C2: `evaluate_series` accepts "train" and "val". Nothing here can reach the test
window; `config.TEST_SPLIT` is not imported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from harness import config
from harness.builder import build, load_indicators, nan_ratio
from harness.legacy import (
    LEGACY_COST_RATE,
    LEGACY_HOLDING_PERIOD,
    LEGACY_MIN_PERIODS_ROLLING,
    LEGACY_ROLLING_WINDOW,
    LEGACY_SMOOTH_WINDOW,
    LEGACY_TRADING_DAYS_PER_YEAR,
    backtest_position,
    load_au9995,
    position_rolling_percentage,
    postprocess_position,
)
from harness.spec import FactorSpec, window_grid

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when a series cannot be evaluated."""


# ======================================================================
# C3 guard — our mirrored constants must match the read-only upstream
# ======================================================================
# config.py restates the backtest conventions so harness code never hardcodes
# them. If upstream ever changes, this fails loudly at import rather than
# silently producing numbers that no longer match the published run.

_MIRRORED = {
    "COST_RATE": (config.COST_RATE, LEGACY_COST_RATE),
    "HOLDING_PERIOD": (config.HOLDING_PERIOD, LEGACY_HOLDING_PERIOD),
    "SMOOTH_WINDOW": (config.SMOOTH_WINDOW, LEGACY_SMOOTH_WINDOW),
    "TRADING_DAYS_PER_YEAR": (config.TRADING_DAYS_PER_YEAR, LEGACY_TRADING_DAYS_PER_YEAR),
    "POSITION_ROLLING_WINDOW": (config.POSITION_ROLLING_WINDOW, LEGACY_ROLLING_WINDOW),
    "POSITION_MIN_PERIODS": (config.POSITION_MIN_PERIODS, LEGACY_MIN_PERIODS_ROLLING),
}
_drift = {k: v for k, v in _MIRRORED.items() if v[0] != v[1]}
if _drift:  # pragma: no cover - fires only if upstream changes
    raise ImportError(
        f"harness/config.py has drifted from the read-only position_mapping.py: "
        f"{ {k: f'config={a} upstream={b}' for k, (a, b) in _drift.items()} }"
    )


# ======================================================================
# The traded asset
# ======================================================================

_RETURNS: pd.Series | None = None


def load_returns(cutoff: str | None = None) -> pd.Series:
    """Au9995 daily simple return, truncated at the data cutoff.

    `r_t = (P_t - P_{t-1}) / P_{t-1}`. Combined with the `pos.shift(1)` inside
    `backtest_position`, this is T+1 execution: yesterday's signal earns today's
    return.
    """
    global _RETURNS
    if _RETURNS is None:
        price = load_au9995(str(config.AU9995_PATH))
        _RETURNS = price.pct_change().rename("ret")
    cut = pd.Timestamp(cutoff or config.DATA_CUTOFF)
    return _RETURNS[_RETURNS.index <= cut]


def clear_cache() -> None:
    global _RETURNS
    _RETURNS = None


# ======================================================================
# Direction
# ======================================================================


def fit_direction(series: pd.Series, returns: pd.Series | None = None) -> int:
    """Trade direction (+1 / -1) from the Spearman IC, fitted on **train only**.

    Upstream's `compute_ic_direction` fits this on the full sample. That is the
    single biggest reason CLAUDE.md A7 forbids treating the published Sharpe as
    a target: a sign chosen with hindsight is worth real basis points, and it
    would leak straight into the validation window.

    Here the sign is decided on 2015–2021 and then frozen. Whatever it does to
    2022–2023 is honest out-of-sample behaviour.

    Returns +1 when the IC is NaN or exactly zero, matching upstream's
    `direction or 1` fallback, so a degenerate factor is merely useless rather
    than silently inverted.
    """
    if returns is None:
        returns = load_returns()
    lo, hi = config.TRAIN_SPLIT
    aligned = pd.concat([series.loc[lo:hi], returns.loc[lo:hi]], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(aligned) < 30:
        return 1
    ic = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method="spearman")
    if pd.isna(ic) or ic == 0:
        return 1
    return int(np.sign(ic))


# ======================================================================
# Positions
# ======================================================================


def _positions(series: pd.Series, direction: int,
               calendar: pd.Index | None = None) -> pd.Series:
    """Factor series -> post-processed position, on the **full** history.

    Order mirrors `position_mapping.main` exactly: map, apply direction, then
    `postprocess_position` (holding-period sparsify, then TWAP smooth).

    Why map on the full series and slice afterwards, rather than slicing first:
    `position_rolling_percentage` needs a 252-day rolling history (60 minimum).
    Slice-then-map would leave the first year of every split without a position
    — and the validation window is only two years long, so it would silently
    throw away half the out-of-sample evidence. Mapping first is not look-ahead:
    every operation in the chain is strictly backward-looking, so a position on
    2022-01-03 uses 2021 data exactly as it would have live.

    Why reindex onto a shared `calendar` first: `apply_holding_period` picks
    rebalance days as `valid_idx[::10]`, counted from each series' own first
    valid row. Left alone, a factor whose indicator starts in 2004 rebalances
    on different days from one whose indicator starts in 1991 — the trading
    calendar would become a function of data vintage. Measured on pmf_104203,
    that artefact alone moves the Sharpe by 0.011. Pinning every factor to the
    traded asset's calendar removes it and is also simply correct: a position
    on a day Au9995 does not trade is not a position.
    """
    if calendar is not None:
        series = series.reindex(calendar)
    mapped = position_rolling_percentage(series.replace([np.inf, -np.inf], np.nan))
    return postprocess_position(mapped * direction)


# ======================================================================
# Metrics over a date range
# ======================================================================


@dataclass(frozen=True)
class SplitMetrics:
    """One (series, direction, date range) evaluated."""

    split: str
    sharpe: float
    ann_return: float
    ann_vol: float
    ann_turnover: float
    max_drawdown: float
    calmar: float
    win_rate: float
    n_days: int

    @property
    def sharpe_or_zero(self) -> float:
        """NaN Sharpe counts as zero, matching upstream's portfolio step."""
        return 0.0 if pd.isna(self.sharpe) else float(self.sharpe)

    def as_dict(self) -> dict:
        return {
            "split": self.split, "sharpe": self.sharpe, "ann_return": self.ann_return,
            "ann_vol": self.ann_vol, "ann_turnover": self.ann_turnover,
            "max_drawdown": self.max_drawdown, "calmar": self.calmar,
            "win_rate": self.win_rate, "n_days": self.n_days,
        }


def _metrics_over(position: pd.Series, returns: pd.Series,
                  lo: str | None, hi: str | None, label: str) -> SplitMetrics:
    pos = position.loc[lo:hi]
    _, _, _, m = backtest_position(pos, returns)
    return SplitMetrics(
        split=label,
        sharpe=float(m["Sharpe"]) if not pd.isna(m["Sharpe"]) else float("nan"),
        ann_return=float(m["ann_return"]) if not pd.isna(m["ann_return"]) else float("nan"),
        ann_vol=float(m["ann_vol"]) if not pd.isna(m["ann_vol"]) else float("nan"),
        ann_turnover=float(m["ann_turnover"]) if not pd.isna(m["ann_turnover"]) else float("nan"),
        max_drawdown=float(m["max_drawdown"]) if not pd.isna(m["max_drawdown"]) else float("nan"),
        calmar=float(m["Calmar"]) if not pd.isna(m["Calmar"]) else float("nan"),
        win_rate=float(m["win_rate"]) if not pd.isna(m["win_rate"]) else float("nan"),
        n_days=int(pos.dropna().index.intersection(returns.dropna().index).size),
    )


def evaluate_series(series: pd.Series, split: str,
                    direction: int | None = None,
                    returns: pd.Series | None = None) -> SplitMetrics:
    """Backtest one factor series over one split. **C2 enforced here.**

    `split` is "train" or "val". Passing "test" raises; the test window is
    reachable only through `scripts/unlock_test.py`. This is an architectural
    constraint, not a discipline — see CLAUDE.md C2.
    """
    if split not in config.ALLOWED_SPLITS:
        raise ValueError(
            f'split must be one of {list(config.ALLOWED_SPLITS)}, got "{split}". '
            f"The test window is locked (CLAUDE.md C2) and reachable only via "
            f"scripts/unlock_test.py --i-understand-this-is-one-shot."
        )
    if returns is None:
        returns = load_returns()
    if direction is None:
        direction = fit_direction(series, returns)
    lo, hi = config.SPLITS[split]
    position = _positions(series, direction, calendar=returns.index)
    return _metrics_over(position, returns, lo, hi, split)


def reproduce_legacy_metrics(series: pd.Series, direction: int) -> SplitMetrics:
    """Reproduce the published in-sample run — **verification only**.

    `results/single_factor_metrics.csv` was produced over the full history
    through 2023-12-31 with a full-sample direction. tests/test_phase2.py calls
    this to prove the wrapper reproduces upstream bit-for-bit.

    This is not a C2 hole: `config.LEGACY_CUTOFF_DATE` is 2023-12-31, the last
    day of the validation window, so no test-window data is reachable from here
    either. It exists so that `evaluate_series` can keep a hard guard instead of
    growing an "except when verifying" branch.

    A7: the numbers this returns are in-sample and must never be a target.
    """
    returns = load_returns(cutoff=config.LEGACY_CUTOFF_DATE)
    position = _positions(series, direction)
    return _metrics_over(position, returns, None, config.LEGACY_CUTOFF_DATE, "legacy")


# ======================================================================
# Window-grid scan
# ======================================================================


@dataclass(frozen=True)
class CellResult:
    windows: dict[str, int]
    direction: int
    nan_ratio: float
    train: SplitMetrics
    val: SplitMetrics
    passed: bool
    fail_reasons: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "windows": dict(self.windows), "direction": self.direction,
            "nan_ratio": round(self.nan_ratio, 4),
            "sharpe_train": round(self.train.sharpe_or_zero, 4),
            "sharpe_val": round(self.val.sharpe_or_zero, 4),
            "ann_turnover": round(self.val.ann_turnover, 4)
            if not pd.isna(self.val.ann_turnover) else None,
            "passed": self.passed, "fail_reasons": list(self.fail_reasons),
        }


@dataclass(frozen=True)
class ScanResult:
    """Everything the Analysis Unit needs, and nothing it must not see."""

    spec_hash: str
    structure_label: str
    n_cells: int
    sharpe_median_train: float
    sharpe_median_val: float
    sharpe_iqr_val: float
    pass_rate: float
    sharpe_max_val: float          # C5: recorded, never used for a decision
    ann_turnover_median: float
    nan_ratio_median: float
    grid_detail: tuple[CellResult, ...] = field(repr=False)

    @property
    def fitness(self) -> float:
        """plan.md §4: the conservative estimate, not the optimistic one."""
        return min(self.sharpe_median_train, self.sharpe_median_val)

    def as_dict(self) -> dict:
        return {
            "spec_hash": self.spec_hash, "structure_label": self.structure_label,
            "n_cells": self.n_cells,
            "sharpe_median_train": round(self.sharpe_median_train, 4),
            "sharpe_median_val": round(self.sharpe_median_val, 4),
            "sharpe_iqr_val": round(self.sharpe_iqr_val, 4),
            "pass_rate": round(self.pass_rate, 4),
            "sharpe_max_val": round(self.sharpe_max_val, 4),
            "ann_turnover_median": round(self.ann_turnover_median, 4),
            "nan_ratio_median": round(self.nan_ratio_median, 4),
            "fitness": round(self.fitness, 4),
            "grid_detail": [c.as_dict() for c in self.grid_detail],
        }


def _cell_verdict(train: SplitMetrics, val: SplitMetrics) -> tuple[bool, tuple[str, ...]]:
    """Does this one grid cell pass? GATE_1 / GATE_2 / GATE_4 only.

    See config.CELL_PASS_GATES for why the other three are excluded.

    GATE_2 (decay ratio) is only meaningful when the train Sharpe is positive.
    A cell that lost money in sample is failed on GATE_2 rather than being
    handed a flattering negative-over-negative ratio.
    """
    reasons: list[str] = []
    s_tr, s_va = train.sharpe_or_zero, val.sharpe_or_zero

    if not (s_va > config.GATE_1_MIN_SHARPE_VAL):
        reasons.append("GATE_1")
    if s_tr <= 0:
        reasons.append("GATE_2")
    elif not (s_va / s_tr > config.GATE_2_MIN_DECAY_RATIO):
        reasons.append("GATE_2")
    turn = val.ann_turnover
    if pd.isna(turn) or not (turn < config.GATE_4_TURNOVER_CAP):
        reasons.append("GATE_4")

    return (not reasons), tuple(reasons)


def scan(spec: FactorSpec, inds: Mapping[str, pd.Series] | None = None,
         returns: pd.Series | None = None) -> ScanResult:
    """Scan a spec's whole window grid and summarise its robustness.

    The headline number is `pass_rate`: in how many corners of the window grid
    does this structure survive. A structure that works in 2 cells out of 27 is
    a parameter accident; one that works in 20 is a real effect. That
    distinction, not the peak Sharpe, is what the agent is scored on (C5).
    """
    if inds is None:
        inds = load_indicators(spec.indicators)
    if returns is None:
        returns = load_returns()

    cells: list[CellResult] = []
    for windows in window_grid(spec):
        series = build(spec, windows, inds)
        direction = fit_direction(series, returns)
        position = _positions(series, direction, calendar=returns.index)
        train = _metrics_over(position, returns, *config.TRAIN_SPLIT, "train")
        val = _metrics_over(position, returns, *config.VAL_SPLIT, "val")
        passed, reasons = _cell_verdict(train, val)
        cells.append(CellResult(
            windows=dict(windows), direction=direction,
            nan_ratio=nan_ratio(series), train=train, val=val,
            passed=passed, fail_reasons=reasons,
        ))

    if not cells:  # pragma: no cover - window_grid never returns empty
        raise ValidationError(f"spec {spec.spec_hash[:12]} produced an empty window grid")

    tr = np.array([c.train.sharpe_or_zero for c in cells], dtype=float)
    va = np.array([c.val.sharpe_or_zero for c in cells], dtype=float)
    turn = np.array([c.val.ann_turnover for c in cells], dtype=float)
    nans = np.array([c.nan_ratio for c in cells], dtype=float)

    return ScanResult(
        spec_hash=spec.spec_hash,
        structure_label=spec.structure_label,
        n_cells=len(cells),
        sharpe_median_train=float(np.median(tr)),
        sharpe_median_val=float(np.median(va)),
        sharpe_iqr_val=float(np.percentile(va, 75) - np.percentile(va, 25)),
        pass_rate=float(np.mean([c.passed for c in cells])),
        sharpe_max_val=float(np.max(va)),
        ann_turnover_median=float(np.nanmedian(turn)) if not np.all(np.isnan(turn)) else float("nan"),
        nan_ratio_median=float(np.median(nans)),
        grid_detail=tuple(cells),
    )


# ======================================================================
# Signal-level dedup (GATE_5 input)
# ======================================================================


def signal_ic(a: pd.Series, b: pd.Series,
              lo: str | None = None, hi: str | None = None) -> float:
    """Spearman correlation between two factor series over the evaluation span.

    Spearman rather than Pearson because the position mapping is itself a
    rolling percentile — two factors that rank the days identically produce
    identical positions no matter how different their raw magnitudes look.
    Returns the signed correlation; callers that care about redundancy should
    take the absolute value, since a perfectly inverted factor is just as
    redundant once `fit_direction` flips its sign.
    """
    lo = lo or config.TRAIN_SPLIT[0]
    hi = hi or config.VAL_SPLIT[1]
    frame = pd.concat([a.loc[lo:hi], b.loc[lo:hi]], axis=1).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(frame) < 30:
        return float("nan")
    ic = frame.iloc[:, 0].corr(frame.iloc[:, 1], method="spearman")
    return float(ic) if not pd.isna(ic) else float("nan")


def max_ic_vs_sota(series: pd.Series,
                   sota: Sequence[pd.Series] | Mapping[str, pd.Series]) -> float:
    """Largest |Spearman IC| against any series already in the SOTA set.

    Returns 0.0 for an empty SOTA set — the first accepted factor cannot be
    redundant with anything.
    """
    items = list(sota.values()) if isinstance(sota, Mapping) else list(sota)
    if not items:
        return 0.0
    ics = [abs(signal_ic(series, other)) for other in items]
    ics = [x for x in ics if not np.isnan(x)]
    return float(max(ics)) if ics else 0.0


__all__ = [
    "CellResult",
    "ScanResult",
    "SplitMetrics",
    "ValidationError",
    "clear_cache",
    "evaluate_series",
    "fit_direction",
    "load_returns",
    "max_ic_vs_sota",
    "reproduce_legacy_metrics",
    "scan",
    "signal_ic",
]
