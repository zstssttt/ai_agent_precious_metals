"""Phase 2 acceptance — Validation Unit.

plan.md asks for four things:
  1. evaluate_series matches single_factor_metrics.csv
  2. split="test" raises
  3. the window grid separates fragile from robust
  4. resubmitting a known factor returns IC ~ 1.0

Claim 1 is split in two, because the two halves can fail for different reasons
and conflating them would hide a real bug:
  1a. the backtest *wrapper* is faithful  — feed the published factor column in,
      demand the published metrics back out, exactly;
  1b. the *built* series reproduces them — feed harness-built values in, allow
      two decimals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harness import config
from harness.builder import build, load_indicators
from harness.l0_catalog import L0_SPECS, L0_WINDOWS
from harness.validator import (
    _positions,
    evaluate_series,
    fit_direction,
    load_returns,
    max_ic_vs_sota,
    scan,
    signal_ic,
)

PUBLISHED_VALUES = config.LEGACY_RESULTS / "factor_values_top15.csv"
PUBLISHED_METRICS = config.LEGACY_RESULTS / "single_factor_metrics.csv"


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture(scope="module")
def published_values() -> pd.DataFrame:
    df = pd.read_csv(PUBLISHED_VALUES, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


@pytest.fixture(scope="module")
def published_metrics() -> pd.DataFrame:
    df = pd.read_csv(PUBLISHED_METRICS, encoding="utf-8-sig")
    df = df[df["position_type"] == config.POSITION_TYPE]
    return df.set_index("factor")


@pytest.fixture(scope="module")
def indicators() -> dict:
    return load_indicators()


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return load_returns()


@pytest.fixture(scope="module")
def legacy_runs(published_values, published_metrics, indicators) -> dict:
    """Run every L0 factor through the legacy-reproduction path, once.

    Two variants per factor so the parametrized tests below are free:
    `from_csv` feeds the published column, `from_build` feeds our own.
    """
    from harness.validator import reproduce_legacy_metrics

    out = {}
    for name, spec in L0_SPECS.items():
        direction = int(published_metrics.loc[name, "direction"]) or 1
        built = build(spec, L0_WINDOWS[name], indicators).reindex(published_values.index)
        out[name] = {
            "direction": direction,
            "from_csv": reproduce_legacy_metrics(published_values[name], direction),
            "from_build": reproduce_legacy_metrics(built, direction),
        }
    return out


def _agree(got: float, want: float, tol: float) -> bool:
    if pd.isna(got) and pd.isna(want):
        return True
    if pd.isna(got) or pd.isna(want):
        return False
    return abs(got - want) <= tol


# ======================================================================
# 1a. The wrapper is faithful — exact, no tolerance
# ======================================================================


@pytest.mark.parametrize("name", sorted(L0_SPECS))
def test_wrapper_reproduces_published_metrics_exactly(name, legacy_runs, published_metrics):
    """Published factor column in -> published metrics out, to the last digit.

    Any drift here means the harness changed the backtest, not the factor.
    """
    got = legacy_runs[name]["from_csv"]
    want = published_metrics.loc[name]
    for field, column in [("sharpe", "Sharpe"), ("ann_turnover", "年化换手率"),
                          ("ann_return", "年化收益率"), ("max_drawdown", "最大回撤")]:
        assert _agree(getattr(got, field), float(want[column]), 1e-9), (
            f"{name}.{field}: wrapper {getattr(got, field)} vs published {want[column]}"
        )


# ======================================================================
# 1b. The built series reproduces them — two decimals (plan.md Phase 2)
# ======================================================================


@pytest.mark.parametrize("name", sorted(L0_SPECS))
def test_built_series_reproduces_published_sharpe_to_two_decimals(
    name, legacy_runs, published_metrics
):
    """Harness-built values in -> published Sharpe out, to 2 d.p.

    Not exact, and the gap is understood rather than tolerated:
    `position_rolling_percentage` ranks with a strict `<`, so a 1e-17 difference
    in the last bit of a factor value can flip one comparison and move the
    Sharpe in the third decimal. pmf_055491 does exactly that (0.725 vs 0.726).
    """
    got = legacy_runs[name]["from_build"].sharpe
    want = float(published_metrics.loc[name, "Sharpe"])
    assert _agree(got, want, 0.005), f"{name}: built {got} vs published {want}"


# ======================================================================
# 2. C2 — the test window is unreachable
# ======================================================================


def test_evaluate_series_rejects_the_test_split(published_values):
    series = published_values["pmf_051056"]
    with pytest.raises(ValueError, match="test window is locked"):
        evaluate_series(series, "test")


@pytest.mark.parametrize("bad", ["Test", "TEST", "oos", "full", "", "train "])
def test_evaluate_series_rejects_anything_not_whitelisted(bad, published_values):
    with pytest.raises(ValueError):
        evaluate_series(published_values["pmf_051056"], bad)


@pytest.mark.parametrize("module", ["validator", "builder", "spec", "journal"])
def test_no_harness_module_reads_the_test_split(module):
    """Structural guard: TEST_SPLIT may be named in prose, never in code.

    Parsed rather than grepped, so a docstring explaining the rule does not
    trip it and an `import` hidden inside a function still does.
    """
    import ast

    path = config.PROJECT_ROOT / "harness" / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Attribute) and node.attr == "TEST_SPLIT")
        or (isinstance(node, ast.Name) and node.id == "TEST_SPLIT")
        or (isinstance(node, ast.alias) and node.name == "TEST_SPLIT")
    ]
    assert not offenders, (
        f"harness/{module}.py references TEST_SPLIT at line(s) {offenders}; "
        f"the test window is reachable only from scripts/unlock_test.py (C2)"
    )


def test_split_metrics_cover_only_the_requested_window(published_values, returns):
    series = published_values["pmf_051056"]
    train = evaluate_series(series, "train", returns=returns)
    val = evaluate_series(series, "val", returns=returns)
    # 7 years vs 2 years, at roughly 243 trading days a year.
    assert 1500 < train.n_days < 1800
    assert 400 < val.n_days < 550


# ======================================================================
# 3. The grid separates fragile from robust
# ======================================================================


@pytest.fixture(scope="module")
def scans(indicators, returns) -> dict:
    """Scan a handful of L0 specs. Not all 15 — each costs ~6s."""
    names = ["pmf_051056", "pmf_131810", "pmf_066409"]
    return {n: scan(L0_SPECS[n], indicators, returns) for n in names}


def test_scan_covers_the_whole_window_grid(scans):
    from harness.spec import window_grid

    for name, result in scans.items():
        assert result.n_cells == len(window_grid(L0_SPECS[name]))
        assert len(result.grid_detail) == result.n_cells
        assert 0.0 <= result.pass_rate <= 1.0


def test_scan_is_deterministic(indicators, returns):
    a = scan(L0_SPECS["pmf_131810"], indicators, returns)
    b = scan(L0_SPECS["pmf_131810"], indicators, returns)
    assert a.as_dict() == b.as_dict()


def test_grid_distinguishes_fragile_from_median(scans):
    """A fragile structure shows a high peak over a weak middle.

    pmf_051056 is the live example: in-sample over the full history it is a
    0.466-Sharpe factor that made the published portfolio, but scanned over the
    train/val split its best cell reaches 0.75 while the median of its nine
    cells is negative. That gap is precisely what pass_rate exists to expose,
    and precisely why C5 forbids deciding on sharpe_max.
    """
    fragile = scans["pmf_051056"]
    assert fragile.sharpe_max_val > 0.5
    assert fragile.sharpe_median_val < fragile.sharpe_max_val
    assert fragile.pass_rate < config.GATE_3_MIN_PASS_RATE
    # fitness must follow the median, never the peak
    assert fragile.fitness == min(fragile.sharpe_median_train, fragile.sharpe_median_val)
    assert fragile.fitness < fragile.sharpe_max_val


def test_sharpe_max_never_feeds_fitness(scans):
    """C5 as an executable statement, not a comment."""
    import dataclasses

    original = scans["pmf_131810"]
    tampered = dataclasses.replace(original, sharpe_max_val=original.sharpe_max_val + 99.0)
    assert tampered.fitness == original.fitness


def test_cell_pass_predicate_excludes_the_circular_gate():
    assert "GATE_3" not in config.CELL_PASS_GATES
    assert set(config.CELL_PASS_GATES) == {"GATE_1", "GATE_2", "GATE_4"}


def test_failing_cells_carry_a_reason(scans):
    for result in scans.values():
        for cell in result.grid_detail:
            assert cell.passed == (not cell.fail_reasons)
            assert set(cell.fail_reasons) <= set(config.CELL_PASS_GATES)


# ======================================================================
# 4. Signal-level dedup
# ======================================================================


def test_a_factor_is_perfectly_redundant_with_itself(published_values):
    series = published_values["pmf_051056"]
    assert max_ic_vs_sota(series, [series]) == pytest.approx(1.0, abs=1e-9)


def test_an_inverted_factor_is_just_as_redundant(published_values):
    """fit_direction will flip the sign anyway, so -x carries the same signal."""
    series = published_values["pmf_051056"]
    assert max_ic_vs_sota(series, [-series]) == pytest.approx(1.0, abs=1e-9)


def test_empty_sota_set_is_not_redundant(published_values):
    assert max_ic_vs_sota(published_values["pmf_051056"], []) == 0.0


def test_dedup_takes_the_maximum_over_the_sota_set(published_values):
    target = published_values["pmf_051056"]
    others = [published_values[n] for n in ("pmf_131810", "pmf_066409")]
    assert max_ic_vs_sota(target, others + [target]) == pytest.approx(1.0, abs=1e-9)
    assert max_ic_vs_sota(target, others) < config.GATE_5_MAX_IC_VS_SOTA


def test_unrelated_l0_factors_are_not_flagged_as_duplicates(published_values):
    """Sanity: the 15 hand-designed factors are mostly distinct signals."""
    names = sorted(L0_SPECS)
    pairs = [
        abs(signal_ic(published_values[a], published_values[b]))
        for i, a in enumerate(names) for b in names[i + 1:]
    ]
    pairs = [p for p in pairs if not np.isnan(p)]
    assert len(pairs) > 50
    assert np.median(pairs) < 0.3


# ======================================================================
# Design decisions, as executable claims
# ======================================================================


def test_direction_cannot_see_the_validation_window(returns):
    """A series with signal *only* in val must not produce a fitted direction.

    An earlier version of this test perturbed the val window of a real factor
    and asserted the direction held. It passed even with the sabotage of
    fitting on train+val — Spearman is rank-based and train is three and a half
    times longer, so two years of inverted data cannot outvote seven. A test
    that a known-broken implementation still passes is worth nothing.

    This construction discriminates. The series is NaN across all of train, so
    a train-only fit sees fewer than 30 observations and must fall back to +1.
    In val it is the exact negative of the traded return, so any implementation
    that looks there computes an IC of -1 and returns -1.
    """
    series = pd.Series(np.nan, index=returns.index)
    lo, hi = config.VAL_SPLIT
    val_mask = (series.index >= lo) & (series.index <= hi)
    series[val_mask] = -returns[val_mask]

    assert fit_direction(series, returns) == 1, (
        "direction was fitted on data outside the training window — the sign "
        "is being chosen with hindsight (CLAUDE.md A7)"
    )


def test_direction_does_use_the_training_window(returns):
    """The mirror of the test above: it must still work when train has signal."""
    series = pd.Series(np.nan, index=returns.index)
    lo, hi = config.TRAIN_SPLIT
    train_mask = (series.index >= lo) & (series.index <= hi)
    series[train_mask] = -returns[train_mask]

    assert fit_direction(series, returns) == -1


def test_direction_is_unchanged_when_validation_data_is_removed(published_values, returns):
    """Same claim from the other side, on a real factor."""
    series = published_values["pmf_051056"]
    lo, hi = config.VAL_SPLIT
    blanked = series.copy()
    blanked[(blanked.index >= lo) & (blanked.index <= hi)] = np.nan
    assert fit_direction(blanked, returns) == fit_direction(series, returns)


def test_mapping_before_slicing_keeps_the_whole_validation_window(
    published_values, returns
):
    """Slice-then-map would blank the first ~60 trading days of val.

    The validation window is only two years long, so warming the rolling
    percentile up inside it would discard a large share of the out-of-sample
    evidence. This test pins the chosen order by comparing against the
    alternative.
    """
    series = published_values["pmf_051056"]
    lo, hi = config.VAL_SPLIT

    chosen = _positions(series, 1, calendar=returns.index).loc[lo:hi]
    alternative = _positions(series.loc[lo:hi], 1)

    assert chosen.iloc[0] == chosen.iloc[0], "first val day has no position"
    assert not pd.isna(chosen.iloc[0])
    assert pd.isna(alternative.iloc[0])
    assert chosen.notna().sum() > alternative.notna().sum() + 50


def test_calendar_pinning_makes_rebalance_days_shared(indicators, returns):
    """Every factor must rebalance on the same days, whatever its data vintage.

    pmf_104203 is built on an indicator starting 2004-11-25; pmf_051056 on one
    starting 1991. Without pinning, `apply_holding_period` counts every tenth
    row from each series' own first valid day and the two drift apart — worth
    0.011 of Sharpe on pmf_104203 alone.
    """
    late = build(L0_SPECS["pmf_104203"], L0_WINDOWS["pmf_104203"], indicators)
    early = build(L0_SPECS["pmf_051056"], L0_WINDOWS["pmf_051056"], indicators)

    pinned = [_positions(s, 1, calendar=returns.index) for s in (late, early)]
    assert pinned[0].index.equals(pinned[1].index)

    unpinned = [_positions(s, 1) for s in (late, early)]
    assert not unpinned[0].index.equals(unpinned[1].index)


def test_mirrored_constants_match_upstream():
    """config.py restates the backtest conventions; they must not drift (C3)."""
    from harness import legacy

    assert config.COST_RATE == legacy.LEGACY_COST_RATE
    assert config.HOLDING_PERIOD == legacy.LEGACY_HOLDING_PERIOD
    assert config.SMOOTH_WINDOW == legacy.LEGACY_SMOOTH_WINDOW
    assert config.TRADING_DAYS_PER_YEAR == legacy.LEGACY_TRADING_DAYS_PER_YEAR
    assert config.POSITION_ROLLING_WINDOW == legacy.LEGACY_ROLLING_WINDOW
    assert config.POSITION_MIN_PERIODS == legacy.LEGACY_MIN_PERIODS_ROLLING


def test_returns_are_truncated_at_the_data_cutoff(returns):
    assert returns.index.max() <= pd.Timestamp(config.DATA_CUTOFF)
