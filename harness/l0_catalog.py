"""L0 — the 15 hand-designed factors, re-expressed as v4 specs.

Source of truth: the 15 ``factor_*`` functions in the read-only
``final_src_github/src/factors/build_factors.py`` (C3 — imported, never edited).
Each entry here was transcribed line by line from that file, not from the
README summaries.

Two jobs:

1. **Schema coverage proof.** ``tests/test_phase0.py`` constructs all 15. If any
   one of them cannot be expressed, the spec space does not cover the human
   designs and Phase 1's kill criterion has fired.
2. **Benchmark B2.** plan.md §4 L3 re-runs these 15 on the harness's own
   train/val split as the human baseline, and plan.md §1 counts how many of
   them the agent rediscovers.

``L0_WINDOWS`` records the window values the original formulas used. Those are
deliberately NOT part of the specs (A4) — they are the single grid cell the
human happened to pick, and Phase 2 scans the whole grid around them.
"""

from __future__ import annotations

from harness.spec import FactorSpec, SpecInput, SpecLeg


def _i(indicator: str, *pre: str, pre_windows: tuple[str, ...] = ()) -> SpecInput:
    return SpecInput(indicator=indicator, pre=tuple(pre), pre_windows=pre_windows)


def _leg(op: str, *inputs: SpecInput, windows: tuple[str, ...] = ()) -> SpecLeg:
    return SpecLeg(op=op, inputs=tuple(inputs), op_windows=windows)


# ======================================================================
# Theme 1 — dollar
# ======================================================================

pmf_128453 = FactorSpec(
    theme="dollar",
    template="risk_adjusted",
    legs={
        "numerator": _leg(
            "corr_change",
            _i("PMF0012", "diff", "rank", pre_windows=("w1",)),
            _i("PMF0036", "diff", "rank", pre_windows=("w1",)),
            windows=("w1", "w1"),
        ),
        "risk": _leg("level", _i("PMF0007")),
    },
    hypothesis=(
        "When the dollar and SPDR cash flow start co-moving differently, gold is "
        "being repriced by the currency channel rather than by flow; scale that "
        "signal down when geopolitical risk is already elevated and dominating."
    ),
)

pmf_011095 = FactorSpec(
    theme="dollar",
    template="identity",
    legs={
        "value": _leg(
            "corr_slope",
            _i("PMF0012", "diff", "rank", pre_windows=("w1",)),
            _i("PMF0031", "pct_ret", "rank", pre_windows=("w1",)),
            windows=("w1", "w1"),
        ),
    },
    hypothesis=(
        "A rising trend in dollar/SPDR-flow co-movement means the currency channel "
        "is strengthening its grip on ETF demand; the slope leads the level."
    ),
)

pmf_069545 = FactorSpec(
    theme="dollar",
    template="direction_x_intensity",
    legs={
        "direction": _leg("ma_cross", _i("PMF0012", "diff"), windows=("w2", "w3")),
        "intensity": _leg("abs_zscore", _i("PMF0042"), windows=("w1",)),
    },
    hypothesis=(
        "Dollar trend sets the sign of gold's opportunity cost; an inflation print "
        "far from its recent mean says how forcefully that channel should bite."
    ),
)


# ======================================================================
# Theme 2 — goldprice
# ======================================================================

pmf_051113 = FactorSpec(
    theme="goldprice",
    template="direction_x_intensity",
    legs={
        "direction": _leg("ma_diff", _i("PMF0007"), windows=("w1", "w1")),
        "intensity": _leg("ma_spread_abs", _i("PMF0020", "pct_ret"), windows=("w1", "w2")),
    },
    hypothesis=(
        "Escalating geopolitical risk sets the direction of safe-haven demand; the "
        "on/offshore basis pulling away from its own mean measures how dislocated "
        "physical pricing has become."
    ),
)

pmf_127111 = FactorSpec(
    theme="goldprice",
    template="risk_adjusted",
    legs={
        "numerator": _leg(
            "corr_change",
            _i("PMF0011", "diff", "rank", pre_windows=("w1",)),
            _i("PMF0031", "diff", "rank", pre_windows=("w1",)),
            windows=("w1", "w1"),
        ),
        "risk": _leg("level", _i("PMF0018")),
    },
    hypothesis=(
        "A shift in how Shanghai physical gold tracks SPDR flow is informative, but "
        "when implied volatility is high everything co-moves, so shrink the signal."
    ),
)

pmf_125861 = FactorSpec(
    theme="goldprice",
    template="risk_adjusted",
    legs={
        "numerator": _leg(
            "corr_change",
            _i("PMF0004", "diff", "rank", pre_windows=("w1",)),
            _i("PMF0036", "diff", "rank", pre_windows=("w1",)),
            windows=("w1", "w1"),
        ),
        "risk": _leg("level", _i("PMF0007")),
    },
    hypothesis=(
        "Same price-versus-flow co-movement test as pmf_128453 but on the global "
        "COMEX leg rather than the dollar, normalised by the prevailing risk regime."
    ),
)


# ======================================================================
# Theme 3 — etf
# ======================================================================

pmf_131810 = FactorSpec(
    theme="etf",
    template="risk_adjusted",
    legs={
        "numerator": _leg(
            "corr_change",
            _i("PMF0019", "diff"),
            _i("PMF0001", "pct_ret", "lag1"),
            windows=("w1", "w1"),
        ),
        "risk": _leg("level", _i("PMF0007")),
    },
    hypothesis=(
        "ETF holdings respond to the bond market with a day's lag; when that "
        "response changes, the rates channel is being repriced — discount it by "
        "the geopolitical regime."
    ),
)

pmf_104203 = FactorSpec(
    theme="etf",
    template="signal_x_persistence",
    legs={
        "signal": _leg("aroon_balance", _i("PMF0031", "pct_ret"), windows=("w1",)),
        "persistence": _leg("path_efficiency", _i("PMF0031", "pct_ret"), windows=("w1",)),
    },
    hypothesis=(
        "Aroon on SPDR flow returns says which way fast money is leaning; path "
        "efficiency says whether it walked there in a straight line or zig-zagged, "
        "i.e. whether to trust the lean."
    ),
)

pmf_058341 = FactorSpec(
    theme="etf",
    template="direction_x_intensity",
    legs={
        "direction": _leg("ma_diff", _i("PMF0031"), windows=("w1", "w1")),
        "intensity": _leg(
            "ratio_over_logsize",
            _i("PMF0013", "pct_ret", "abs"),
            _i("PMF0031"),
            windows=("w1",),
        ),
    },
    hypothesis=(
        "Direction from whether fast money is creating or redeeming; intensity from "
        "how violently the curve is moving per unit of ETF size."
    ),
)


# ======================================================================
# Theme 4 — rates
# ======================================================================

pmf_102135 = FactorSpec(
    theme="rates",
    template="signal_x_persistence",
    legs={
        "signal": _leg("ma_cross", _i("PMF0042"), windows=("w1", "w2")),
        "persistence": _leg("autocorr_abs", _i("PMF0042"), windows=("w1",)),
    },
    hypothesis=(
        "Core CPI's short MA crossing its long MA says inflation is accelerating; "
        "its own autocorrelation says whether that acceleration has inertia or is "
        "just a noisy print."
    ),
)

pmf_003678 = FactorSpec(
    theme="rates",
    template="identity",
    legs={
        "value": _leg(
            "corr_change",
            _i("PMF0002", "diff"),
            _i("PMF0019", "diff", "lag1"),
            windows=("w1", "w1"),
        ),
    },
    hypothesis=(
        "Gold's textbook driver is the real yield. When the link between TIPS moves "
        "and ETF holdings changes breaks down, the market's pricing logic is "
        "rotating and the shift itself is tradable."
    ),
)

pmf_118259 = FactorSpec(
    theme="rates",
    template="direction_x_position",
    legs={
        "direction": _leg("ma_cross", _i("PMF0042"), windows=("w2", "w3")),
        "position": _leg("boll_position", _i("PMF0042"), windows=("w1",)),
    },
    hypothesis=(
        "Medium-term inflation trend sets the sign; where CPI sits inside its own "
        "band says how stretched the regime already is."
    ),
)


# ======================================================================
# Theme 5 — risk
# ======================================================================

pmf_051056 = FactorSpec(
    theme="risk",
    template="direction_x_intensity",
    legs={
        "direction": _leg("ma_diff", _i("PMF0007"), windows=("w1", "w1")),
        "intensity": _leg("abs_pct_change", _i("PMF0042"), windows=("w2",)),
    },
    hypothesis=(
        "Geopolitical risk escalating plus inflation moving sharply is the classic "
        "twin driver of safe-haven demand for gold."
    ),
)

pmf_055491 = FactorSpec(
    theme="risk",
    template="direction_x_intensity",
    legs={
        "direction": _leg("ma_diff", _i("PMF0018"), windows=("w2", "w2")),
        "intensity": _leg(
            "ratio_over_logsize",
            _i("PMF0042", "diff", "abs"),
            _i("PMF0036"),
            windows=("w1",),
        ),
    },
    hypothesis=(
        "Rising implied volatility means the market is paying up for protection; "
        "inflation surprise per unit of fund-flow size says how much of that fear "
        "is genuinely macro-driven."
    ),
)

pmf_066409 = FactorSpec(
    theme="risk",
    template="direction_x_intensity",
    legs={
        "direction": _leg("ma_cross", _i("PMF0007"), windows=("w2", "w3")),
        "intensity": _leg(
            "ratio_over_logsize",
            _i("PMF0013", "abs"),
            _i("PMF0031"),
            windows=("w1",),
        ),
    },
    hypothesis=(
        "A slower read on geopolitical risk than pmf_051056, scaled by how extreme "
        "the curve is per unit of ETF flow."
    ),
)


# ======================================================================
# Registry
# ======================================================================

L0_SPECS: dict[str, FactorSpec] = {
    "pmf_128453": pmf_128453,
    "pmf_011095": pmf_011095,
    "pmf_069545": pmf_069545,
    "pmf_051113": pmf_051113,
    "pmf_127111": pmf_127111,
    "pmf_125861": pmf_125861,
    "pmf_131810": pmf_131810,
    "pmf_104203": pmf_104203,
    "pmf_058341": pmf_058341,
    "pmf_102135": pmf_102135,
    "pmf_003678": pmf_003678,
    "pmf_118259": pmf_118259,
    "pmf_051056": pmf_051056,
    "pmf_055491": pmf_055491,
    "pmf_066409": pmf_066409,
}

# The window values each original formula happened to use. Not part of the spec
# (A4); Phase 1 uses them to reproduce factor_values_top15.csv exactly, and
# Phase 2 confirms they sit inside the scanned grid.
L0_WINDOWS: dict[str, dict[str, int]] = {
    "pmf_128453": {"w1": 5},
    "pmf_011095": {"w1": 5},
    "pmf_069545": {"w1": 5, "w2": 20, "w3": 60},
    "pmf_051113": {"w1": 5, "w2": 20},
    "pmf_127111": {"w1": 5},
    "pmf_125861": {"w1": 5},
    "pmf_131810": {"w1": 20},
    "pmf_104203": {"w1": 5},
    "pmf_058341": {"w1": 5},
    "pmf_102135": {"w1": 5, "w2": 20},
    "pmf_003678": {"w1": 20},
    "pmf_118259": {"w1": 5, "w2": 20, "w3": 60},
    "pmf_051056": {"w1": 5, "w2": 20},
    "pmf_055491": {"w1": 5, "w2": 20},
    "pmf_066409": {"w1": 5, "w2": 20, "w3": 60},
}

# The 6 that survived the original study and carry the published portfolio.
# Read off final_src_github/results/portfolio_weights.csv, rows with
# selected == True under position_type == "rolling_percentage" (the style behind
# the README headline table). Their in-sample single-factor Sharpes there are
# 0.454 / 0.140 / 0.457 / 0.466 / 0.726 / 0.133 respectively.
#
# Note the survivor set is style-dependent: the two z-score styles select only 4
# of these (125861 and 051056 drop out). A7 applies — none of these numbers is a
# target, they are in-sample on a different split and must be re-run before any
# comparison.
L0_SURVIVORS: tuple[str, ...] = (
    "pmf_128453",
    "pmf_125861",
    "pmf_131810",
    "pmf_051056",
    "pmf_055491",
    "pmf_066409",
)

__all__ = ["L0_SPECS", "L0_WINDOWS", "L0_SURVIVORS"]
