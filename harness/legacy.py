"""The single import boundary onto the read-only upstream repo (CLAUDE.md C3).

`final_src_github/` is imported, never modified. Every harness module that needs
an upstream primitive imports it from here, so there is exactly one place that
touches `sys.path` and exactly one place to look when the upstream moves.

What we take and why:

* the 8 `ts_*` operators — the harness's 14 operators are compositions of these,
  so a reproduction mismatch can only come from our composition, not from the
  primitives;
* `zero2nan` — its exact placement (wrapping `sign()`, wrapping denominators)
  is load-bearing for reproducing the L0 values;
* the 5 `apply_*` composition templates;
* `load_indicator` / `PMF_TO_FILENAME` — the indicator loader and its file map.

Note for anyone reading the upstream source: the 15 factor functions there do
**not** call the `apply_*` helpers. They inline `1 * direction * intensity`.
The helpers are equivalent for all 15 (every intensity leg is already
non-negative, so the helper's extra `np.abs` is idempotent), and we use the
helpers because they are the documented contract.
"""

from __future__ import annotations

import sys

from harness import config

if str(config.LEGACY_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(config.LEGACY_SRC_ROOT))

from src.backtest.position_mapping import (  # noqa: E402
    COST_RATE as LEGACY_COST_RATE,
    HOLDING_PERIOD as LEGACY_HOLDING_PERIOD,
    MIN_PERIODS_ROLLING as LEGACY_MIN_PERIODS_ROLLING,
    POSITION_SMOOTH_WINDOW as LEGACY_SMOOTH_WINDOW,
    ROLLING_WINDOW as LEGACY_ROLLING_WINDOW,
    TRADING_DAYS_PER_YEAR as LEGACY_TRADING_DAYS_PER_YEAR,
    backtest_position,
    load_au9995,
    position_rolling_percentage,
    postprocess_position,
)
from src.factors.build_factors import (  # noqa: E402
    PMF_TO_FILENAME,
    TOP15_FACTORS,
    apply_direction_x_intensity,
    apply_direction_x_position,
    apply_identity,
    apply_risk_adjusted,
    apply_signal_x_persistence,
    load_indicator,
    ts_aroon_balance,
    ts_corr_binary,
    ts_delay,
    ts_linreg_slope,
    ts_mean,
    ts_rank,
    ts_std,
    ts_sum,
    zero2nan,
)

__all__ = [
    "LEGACY_COST_RATE",
    "LEGACY_HOLDING_PERIOD",
    "LEGACY_MIN_PERIODS_ROLLING",
    "LEGACY_ROLLING_WINDOW",
    "LEGACY_SMOOTH_WINDOW",
    "LEGACY_TRADING_DAYS_PER_YEAR",
    "PMF_TO_FILENAME",
    "TOP15_FACTORS",
    "backtest_position",
    "load_au9995",
    "position_rolling_percentage",
    "postprocess_position",
    "apply_direction_x_intensity",
    "apply_direction_x_position",
    "apply_identity",
    "apply_risk_adjusted",
    "apply_signal_x_persistence",
    "load_indicator",
    "ts_aroon_balance",
    "ts_corr_binary",
    "ts_delay",
    "ts_linreg_slope",
    "ts_mean",
    "ts_rank",
    "ts_std",
    "ts_sum",
    "zero2nan",
]
