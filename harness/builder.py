"""③ Implementation Unit — (spec, windows) -> pd.Series. No LLM (C1).

Every operator here is a composition of the 8 `ts_*` primitives in the
read-only upstream `build_factors.py` (C3, reached via `harness.legacy`). The
expansions were transcribed from the **bodies of the 15 factor functions**, not
from the `apply_*` helper signatures, because the upstream factor functions
inline their composition rather than calling the helpers.

Two placements are load-bearing and easy to get subtly wrong:

* `zero2nan` wraps `np.sign(...)`, so a flat direction becomes NaN, not 0.
  Drop it and a direction × intensity factor silently emits a run of zeros.
* `zero2nan` wraps every denominator, so division by zero yields NaN, not inf.
  GATE_6 counts NaN; it does not catch inf.

Determinism: same (spec, windows) in, same series out. No randomness, no
network, no clock.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from harness import config
from harness.legacy import (
    PMF_TO_FILENAME,
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
from harness.spec import OPERATORS, PRE_TRANSFORMS, FactorSpec, SpecInput, SpecLeg

logger = logging.getLogger(__name__)


class BuildError(ValueError):
    """Raised when a spec cannot be turned into a series."""


# ======================================================================
# Indicator loading
# ======================================================================

_CACHE: dict[str, pd.Series] = {}


def load_indicators(codes: Iterable[str] | None = None,
                    *, cutoff: str | None = None) -> dict[str, pd.Series]:
    """Load indicator series, truncated at the data cutoff.

    The alignment script forward-fills every file out to 2026-12-31; the last
    265 rows of 8 of the 13 files are identical padding with no new
    observation behind them. `config.DATA_CUTOFF` cuts them off here, once, so
    no downstream module has to remember to.

    Truncating the tail is safe for every operator in this module because they
    are all backward-looking rolling windows — a value at date t never depends
    on a row after t.
    """
    cut = pd.Timestamp(cutoff or config.DATA_CUTOFF)
    wanted = list(codes) if codes is not None else list(PMF_TO_FILENAME)

    out: dict[str, pd.Series] = {}
    for code in wanted:
        key = f"{code}@{cut.date()}"
        if key not in _CACHE:
            if code not in PMF_TO_FILENAME:
                raise BuildError(
                    f'indicator "{code}" has no file mapping in PMF_TO_FILENAME'
                )
            series = load_indicator(code, config.ALIGNED_ROOT, PMF_TO_FILENAME)
            _CACHE[key] = series[series.index <= cut]
        out[code] = _CACHE[key]
    return out


def clear_cache() -> None:
    _CACHE.clear()


# ======================================================================
# Pre-transforms
# ======================================================================


def _apply_pre(series: pd.Series, inp: SpecInput, windows: Mapping[str, int]) -> pd.Series:
    """Apply an input's ordered pre-transforms."""
    roles = inp.resolved_pre_windows()
    role_idx = 0
    out = series
    for name in inp.pre:
        spec_pre = PRE_TRANSFORMS[name]
        if name == "diff":
            out = out - ts_delay(out, 1)
        elif name == "pct_ret":
            out = out / zero2nan(ts_delay(out, 1)) - 1
        elif name == "lag1":
            out = ts_delay(out, 1)
        elif name == "abs":
            out = np.abs(out)
        elif name == "log_abs":
            out = np.abs(np.log(zero2nan(np.abs(out))))
        elif name == "rank":
            out = ts_rank(out, windows[roles[role_idx]])
        else:  # pragma: no cover - whitelist is validated upstream
            raise BuildError(f'pre-transform "{name}" has no implementation')
        role_idx += spec_pre.n_windows
    return out


# ======================================================================
# Operators
# ======================================================================
# Each takes the already-pre-transformed input series plus the resolved window
# values for its slots, in slot order.


def _op_level(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    return xs[0]


def _op_ma_diff(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    ma = ts_mean(xs[0], w[0])
    return zero2nan(np.sign(ma - ts_delay(ma, w[1])))


def _op_ma_cross(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    return zero2nan(np.sign(ts_mean(xs[0], w[0]) - ts_mean(xs[0], w[1])))


def _op_ma_spread_abs(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    return np.abs(ts_mean(xs[0], w[0]) - ts_mean(xs[0], w[1]))


def _op_abs_pct_change(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    return np.abs(xs[0] / zero2nan(ts_delay(xs[0], w[0])) - 1)


def _op_abs_zscore(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    x = xs[0]
    return np.abs((x - ts_mean(x, w[0])) / zero2nan(ts_std(x, w[0])))


def _op_linreg_slope(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    return ts_linreg_slope(xs[0], w[0])


def _op_aroon_balance(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    return ts_aroon_balance(xs[0], w[0])


def _op_path_efficiency(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    x, n = xs[0], w[0]
    travelled = ts_sum(np.abs(x - ts_delay(x, 1)), n)
    return np.abs(x - ts_delay(x, n)) / zero2nan(travelled)


def _op_autocorr_abs(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    x = xs[0]
    return np.abs(ts_corr_binary(x, ts_delay(x, 1), w[0]))


def _op_boll_position(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    x, n = xs[0], w[0]
    std = ts_std(x, n)
    lower = ts_mean(x, n) - std
    return (x - lower) / zero2nan(2 * std) - 0.5


def _op_corr_change(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    corr = ts_corr_binary(xs[0], xs[1], w[0])
    return corr - ts_delay(corr, w[1])


def _op_corr_slope(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    return ts_linreg_slope(ts_corr_binary(xs[0], xs[1], w[0]), w[1])


def _op_ratio_over_logsize(xs: Sequence[pd.Series], w: Sequence[int]) -> pd.Series:
    size = np.abs(np.log(zero2nan(np.abs(xs[1]))))
    return ts_mean(xs[0] / zero2nan(size), w[0])


OP_IMPL: dict[str, Callable[[Sequence[pd.Series], Sequence[int]], pd.Series]] = {
    "level": _op_level,
    "ma_diff": _op_ma_diff,
    "ma_cross": _op_ma_cross,
    "ma_spread_abs": _op_ma_spread_abs,
    "abs_pct_change": _op_abs_pct_change,
    "abs_zscore": _op_abs_zscore,
    "linreg_slope": _op_linreg_slope,
    "aroon_balance": _op_aroon_balance,
    "path_efficiency": _op_path_efficiency,
    "autocorr_abs": _op_autocorr_abs,
    "boll_position": _op_boll_position,
    "corr_change": _op_corr_change,
    "corr_slope": _op_corr_slope,
    "ratio_over_logsize": _op_ratio_over_logsize,
}

TEMPLATE_IMPL: dict[str, Callable[..., pd.Series]] = {
    "identity": lambda legs: apply_identity(legs["value"]),
    "risk_adjusted": lambda legs: apply_risk_adjusted(legs["numerator"], legs["risk"]),
    "direction_x_intensity": lambda legs: apply_direction_x_intensity(
        legs["direction"], legs["intensity"]),
    "signal_x_persistence": lambda legs: apply_signal_x_persistence(
        legs["signal"], legs["persistence"]),
    "direction_x_position": lambda legs: apply_direction_x_position(
        legs["direction"], legs["position"]),
}

# Every whitelisted operator and template must have an implementation here.
assert set(OP_IMPL) == set(OPERATORS), set(OPERATORS) ^ set(OP_IMPL)


# ======================================================================
# Build
# ======================================================================


@dataclass(frozen=True)
class BuildResult:
    series: pd.Series
    spec_hash: str
    windows: dict[str, int]

    @property
    def n_obs(self) -> int:
        return int(self.series.notna().sum())


def _build_leg(leg: SpecLeg, inds: Mapping[str, pd.Series],
               windows: Mapping[str, int]) -> pd.Series:
    inputs = [_apply_pre(inds[i.indicator], i, windows) for i in leg.inputs]
    op_window_values = [windows[r] for r in leg.resolved_op_windows()]
    return OP_IMPL[leg.op](inputs, op_window_values)


def build(spec: FactorSpec, windows: Mapping[str, int],
          inds: Mapping[str, pd.Series] | None = None) -> pd.Series:
    """Turn a spec plus one window-grid cell into a factor series.

    `windows` maps every role in `spec.window_roles` to a value from
    `config.WINDOW_VALUES`. It is the validator's job to supply those; the spec
    itself never carries them (A4).
    """
    missing = set(spec.window_roles) - set(windows)
    if missing:
        raise BuildError(
            f"no window value supplied for role(s) {sorted(missing)}; "
            f"this spec needs {list(spec.window_roles)}"
        )

    if inds is None:
        inds = load_indicators(spec.indicators)
    else:
        absent = [c for c in spec.indicators if c not in inds]
        if absent:
            raise BuildError(f"indicator series not supplied: {absent}")

    legs = {role: _build_leg(leg, inds, windows) for role, leg in spec.legs.items()}
    series = TEMPLATE_IMPL[spec.template](legs)
    series = pd.Series(series).sort_index()
    series.name = spec.spec_hash[:12]
    return series


def build_result(spec: FactorSpec, windows: Mapping[str, int],
                 inds: Mapping[str, pd.Series] | None = None) -> BuildResult:
    return BuildResult(
        series=build(spec, windows, inds),
        spec_hash=spec.spec_hash,
        windows=dict(windows),
    )


# ======================================================================
# Coverage diagnostics
# ======================================================================


def nan_ratio(series: pd.Series, start: str | None = None,
              end: str | None = None) -> float:
    """Share of unusable values inside the evaluation window, used by GATE_6.

    Two decisions are baked in here, both backed by measurement on the 15 L0
    factors (see docs/pool_inventory.md):

    **Denominator = rows inside the evaluation window**, not the whole series.
    A factor on SPDR flow has no data before 2004 and one on the on/offshore
    basis none before 2008, but trading starts 2015-01-01. Scored over the full
    index, 11 of the 15 L0 factors breach the 0.3 cap and 5 of the 6 that
    actually made the portfolio get killed — for missing history nobody would
    ever have traded. Scored over the window, exactly 3 breach it, and they are
    the three monthly-CPI-on-a-5-day-window factors that rule R1 warns about;
    no survivor is touched. The gate rediscovers R1 on its own.

    **Numerator counts every non-finite value, NaN and +/-inf alike.** inf is
    real here: a full grid sweep of the 15 L0 specs produces 972 of them, and
    one cell of pmf_102135 is 12% inf inside the window. Counting only NaN
    would wave that through the gate and let it turn a portfolio Sharpe into
    NaN downstream. `zero2nan` guards the denominators this module writes, but
    not every route to inf (see BACKLOG B8).

    An empty window returns 1.0 — maximally unusable — so a factor with no
    overlap fails the gate instead of raising or sneaking through as 0.0.

    `start` / `end` are ISO date strings; both default to the evaluation span
    so a caller who omits them lands on the correct denominator rather than the
    dangerous one.
    """
    lo = start or config.TRAIN_SPLIT[0]
    hi = end or config.VAL_SPLIT[1]
    window = series.loc[lo:hi]
    if len(window) == 0:
        return 1.0
    usable = np.isfinite(window.to_numpy(dtype=float))
    return float(1.0 - usable.mean())


__all__ = [
    "BuildError",
    "BuildResult",
    "OP_IMPL",
    "TEMPLATE_IMPL",
    "build",
    "build_result",
    "clear_cache",
    "load_indicators",
    "nan_ratio",
]
