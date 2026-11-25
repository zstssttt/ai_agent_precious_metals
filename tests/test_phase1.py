"""Phase 1 acceptance — Implementation Unit.

plan.md calls this "the single most important acceptance test in the plan": hand-write specs, build them, and
check the output equals the published factor values pointwise, NaN aligned.

The plan asks for 3 factors. We do all 15, because `harness/l0_catalog.py`
already carries all 15 and a partial pass would hide exactly the kind of
operator mistake this test exists to catch.

Kill criterion (plan.md Phase 1): if a factor cannot be expressed, or the
numbers do not line up, STOP and redesign the schema. Do not tune anything to
make it look right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness import config
from harness.builder import BuildError, build, load_indicators
from harness.l0_catalog import L0_SPECS, L0_WINDOWS
from harness.spec import FactorSpec, SpecInput, SpecLeg

PUBLISHED = config.LEGACY_RESULTS / "factor_values_top15.csv"


@pytest.fixture(scope="module")
def published() -> pd.DataFrame:
    df = pd.read_csv(PUBLISHED, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


@pytest.fixture(scope="module")
def indicators() -> dict:
    return load_indicators()


# ======================================================================
# The reproduction test
# ======================================================================


@pytest.mark.parametrize("name", sorted(L0_SPECS))
def test_l0_factor_reproduces_published_values(name, published, indicators):
    """Pointwise equality against results/factor_values_top15.csv."""
    spec = L0_SPECS[name]
    got = build(spec, L0_WINDOWS[name], indicators)
    want = published[name]

    # Compare on the published index only: the CSV was cut at 2023-12-31 while
    # the harness loads through 2025-11-28. Every operator is backward-looking,
    # so the overlapping range must match exactly.
    got = got.reindex(want.index)

    both_nan = got.isna() & want.isna()
    assert both_nan.sum() + got.notna().sum() >= len(want) * 0.5, (
        f"{name}: suspiciously little overlap"
    )

    # NaN masks must agree exactly — a shifted mask means a window is off by one.
    mismatch_mask = got.isna() != want.isna()
    assert not mismatch_mask.any(), (
        f"{name}: NaN pattern differs on {int(mismatch_mask.sum())} of {len(want)} rows; "
        f"first at {want.index[mismatch_mask][0].date() if mismatch_mask.any() else '-'}"
    )

    common = got.notna()
    assert common.sum() > 0, f"{name}: produced no values at all"
    assert np.allclose(got[common].to_numpy(), want[common].to_numpy(),
                       rtol=1e-9, atol=1e-12), (
        f"{name}: values differ; max abs diff "
        f"{np.nanmax(np.abs(got[common].to_numpy() - want[common].to_numpy()))}"
    )


def test_all_15_are_covered(published):
    assert set(published.columns) == set(L0_SPECS), (
        "the published CSV and the L0 catalogue must describe the same 15 factors"
    )


# ======================================================================
# Determinism and contract
# ======================================================================


def test_build_is_deterministic(indicators):
    spec = L0_SPECS["pmf_051056"]
    a = build(spec, L0_WINDOWS["pmf_051056"], indicators)
    b = build(spec, L0_WINDOWS["pmf_051056"], indicators)
    pd.testing.assert_series_equal(a, b)


def test_different_windows_give_different_series(indicators):
    """The grid has to actually move the output, or scanning it is theatre."""
    spec = L0_SPECS["pmf_051056"]
    base = build(spec, {"w1": 5, "w2": 20}, indicators)
    other = build(spec, {"w1": 60, "w2": 60}, indicators)
    aligned = pd.concat([base, other], axis=1).dropna()
    assert len(aligned) > 100
    assert not np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1])


def test_missing_window_role_raises(indicators):
    with pytest.raises(BuildError, match="no window value supplied"):
        build(L0_SPECS["pmf_051056"], {"w1": 5}, indicators)


def test_data_is_truncated_at_the_cutoff():
    """The ffill padding out to 2026-12-31 must never reach a factor."""
    inds = load_indicators(["PMF0012", "PMF0002"])
    for code, series in inds.items():
        assert series.index.max() <= pd.Timestamp(config.DATA_CUTOFF), code


def test_direction_leg_emits_no_zeros(indicators):
    """zero2nan around sign(): a flat direction is NaN, never 0.

    Drop that wrapper and a direction × intensity factor silently emits runs of
    zeros, which look like "no position" but are really "no information".
    """
    from harness.builder import _build_leg

    spec = L0_SPECS["pmf_051056"]
    direction = _build_leg(spec.legs["direction"], indicators, L0_WINDOWS["pmf_051056"])
    assert (direction.dropna() == 0).sum() == 0
    assert set(np.unique(direction.dropna())) <= {-1.0, 1.0}


def test_build_rejects_unsupplied_indicators():
    spec = L0_SPECS["pmf_051056"]
    with pytest.raises(BuildError, match="indicator series not supplied"):
        build(spec, L0_WINDOWS["pmf_051056"], {})


# ======================================================================
# GATE_6 coverage — waiting on TODO(human) in harness/builder.py
# ======================================================================


def test_nan_ratio_ignores_structurally_absent_history(indicators):
    """A factor must not be punished for history that predates its indicators.

    pmf_051113 uses PMF0020 (on/offshore basis), which begins 2008-01-09, while
    the evaluation window starts 2015-01-01. Measured over the evaluation
    window the series is mostly populated; measured over the full 1990-2025
    index it looks two-thirds empty.
    """
    from harness.builder import nan_ratio

    series = build(L0_SPECS["pmf_051113"], L0_WINDOWS["pmf_051113"], indicators)
    windowed = nan_ratio(series, config.TRAIN_SPLIT[0], config.VAL_SPLIT[1])

    assert 0.0 <= windowed <= 1.0
    assert windowed < config.GATE_6_MAX_NAN_RATIO, (
        f"a surviving L0 factor scores {windowed:.3f} on GATE_6 "
        f"(cap {config.GATE_6_MAX_NAN_RATIO}) — the denominator is wrong"
    )
