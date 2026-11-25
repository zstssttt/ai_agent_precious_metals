"""Phase 3 acceptance — Analysis Unit.

plan.md asks for five things:
  1. an overfit spec (high train, flat val) is stopped by GATE_2
  2. a fragile spec is stopped by GATE_3
  3. a near-copy of an accepted factor is stopped by GATE_5
  4. reject reasons land in knowledge/failures/ and the diagnosis is actionable
  5. the L0 gate table is printed

Most gate logic is tested against **synthetic** ScanResults rather than real
scans. That is deliberate: a real factor exercises one accidental combination
of the six gates, while a synthetic one lets each gate be driven to its
boundary from both sides. The real L0 factors are then run end to end in
`test_l0_gate_table` so the thresholds are also checked against actual data.
"""

from __future__ import annotations

import ast
import dataclasses

import numpy as np
import pytest

from harness import config
from harness.analyst import (
    GATE_ORDER,
    decay_ratio,
    decide,
    diagnose,
    journal_fields,
    judge,
    record_failure,
    run_gates,
)
from harness.builder import load_indicators
from harness.journal import TrialRecord, append_trial, read_trials
from harness.l0_catalog import L0_SPECS
from harness.validator import CellResult, ScanResult, SplitMetrics, load_returns, scan


# ======================================================================
# Synthetic scan results
# ======================================================================


def _metrics(split: str, sharpe: float, turnover: float = 10.0) -> SplitMetrics:
    return SplitMetrics(
        split=split, sharpe=sharpe, ann_return=0.05, ann_vol=0.09,
        ann_turnover=turnover, max_drawdown=0.2, calmar=0.25,
        win_rate=0.5, n_days=500,
    )


def _cell(w1: int, passed: bool, sharpe_val: float = 0.5,
          nan: float = 0.0) -> CellResult:
    return CellResult(
        windows={"w1": w1}, direction=1, nan_ratio=nan,
        train=_metrics("train", 0.5), val=_metrics("val", sharpe_val),
        passed=passed, fail_reasons=() if passed else ("GATE_1",),
    )


def make_scan(*, sharpe_train: float = 0.40, sharpe_val: float = 0.30,
              pass_rate: float = 0.80, sharpe_max: float = 0.90,
              turnover: float = 10.0, nan_ratio: float = 0.0,
              n_cells: int = 27, cells=None) -> ScanResult:
    """A ScanResult with every field dialled independently."""
    if cells is None:
        n_pass = int(round(pass_rate * n_cells))
        cells = tuple(
            _cell(5, i < n_pass, nan=nan_ratio) for i in range(n_cells)
        )
    return ScanResult(
        spec_hash="deadbeef" * 8, structure_label="identity[value=level(PMF0001)]",
        n_cells=n_cells, sharpe_median_train=sharpe_train,
        sharpe_median_val=sharpe_val,
        sharpe_iqr_val=0.2, pass_rate=pass_rate, sharpe_max_val=sharpe_max,
        ann_turnover_median=turnover, nan_ratio_median=nan_ratio,
        grid_detail=cells,
    )


# ======================================================================
# 1. GATE_2 catches overfitting
# ======================================================================


def test_high_train_flat_val_is_caught_by_gate_2():
    result = judge(make_scan(sharpe_train=0.90, sharpe_val=0.10, pass_rate=0.80))
    gates = {g.name: g for g in result.gates}

    assert not gates["GATE_2"].passed
    assert gates["GATE_2"].observed == pytest.approx(0.10 / 0.90)
    assert gates["GATE_1"].passed, "val is positive, GATE_1 should not be the one firing"
    assert gates["GATE_3"].passed
    assert result.verdict == "REFINE"
    assert "overfitting" in result.diagnosis
    assert any("Reduce complexity" in s for s in result.suggestions)


def test_gate_2_passes_when_decay_is_mild():
    result = judge(make_scan(sharpe_train=0.40, sharpe_val=0.30))
    assert {g.name for g in result.gates if not g.passed} == set()
    assert result.verdict == "ACCEPT"


@pytest.mark.parametrize("offset,expected", [(+0.02, True), (0.0, False), (-0.02, False)])
def test_gate_2_boundary_is_strictly_greater_than(offset, expected):
    train = 0.40
    val = train * (config.GATE_2_MIN_DECAY_RATIO + offset)
    gates = {g.name: g for g in run_gates(make_scan(sharpe_train=train, sharpe_val=val))}
    assert gates["GATE_2"].passed is expected


def test_decay_ratio_is_undefined_when_train_lost_money():
    """A fluke that lost in sample must not clear an overfitting gate."""
    assert np.isnan(decay_ratio(-0.05, 0.52))
    assert np.isnan(decay_ratio(0.0, 0.52))

    result = judge(make_scan(sharpe_train=-0.05, sharpe_val=0.52))
    gates = {g.name: g for g in result.gates}
    assert not gates["GATE_2"].passed
    assert "no in-sample performance to decay from" in gates["GATE_2"].note
    assert any("does not make money" in s for s in result.suggestions)
    assert not any("Reduce complexity" in s for s in result.suggestions), (
        "a structure that never worked in sample is not overfit; telling the "
        "agent to simplify it sends it the wrong way"
    )


def test_analyst_and_validator_agree_on_the_decay_rule():
    """Cell level and spec level must not use different definitions."""
    from harness.validator import _cell_verdict

    passed, reasons = _cell_verdict(_metrics("train", -0.05), _metrics("val", 0.52))
    assert "GATE_2" in reasons
    assert np.isnan(decay_ratio(-0.05, 0.52))


# ======================================================================
# 2. GATE_3 catches fragility
# ======================================================================


def test_fragile_spec_is_caught_by_gate_3():
    """2 cells out of 27 work, and the peak is spectacular. Must not pass."""
    result = judge(make_scan(
        sharpe_train=0.40, sharpe_val=0.30,
        pass_rate=2 / 27, sharpe_max=2.50, n_cells=27,
    ))
    gates = {g.name: g for g in result.gates}

    assert not gates["GATE_3"].passed
    assert gates["GATE_1"].passed and gates["GATE_2"].passed
    assert result.verdict == "REFINE"
    assert any("rather than the window" in s for s in result.suggestions)


@pytest.mark.parametrize("offset,expected", [(+0.01, True), (0.0, False), (-0.01, False)])
def test_gate_3_boundary_is_strictly_greater_than(offset, expected):
    """Tests the comparison, not the number — the number moves under calibration."""
    rate = config.GATE_3_MIN_PASS_RATE + offset
    gates = {g.name: g for g in run_gates(make_scan(pass_rate=rate))}
    assert gates["GATE_3"].passed is expected


def test_calibrated_thresholds_match_the_calibration_doc():
    """Pins the values docs/calibration.md justifies.

    The boundary tests above follow config, so they cannot notice a threshold
    silently drifting. This one can. If you change a threshold, change it here
    too and write down why in docs/calibration.md — that is the whole point of
    having the doc.
    """
    assert config.GATE_1_MIN_SHARPE_VAL == 0.0
    assert config.GATE_2_MIN_DECAY_RATIO == 0.5
    assert config.GATE_3_MIN_PASS_RATE == 0.3      # lowered from 0.4 on 2026-11-23
    assert config.GATE_4_TURNOVER_CAP == 25.0
    assert config.GATE_5_MAX_IC_VS_SOTA == 0.70    # tightened from 0.8 on 2026-11-23
    assert config.GATE_6_MAX_NAN_RATIO == 0.3


def test_diagnosis_locates_the_surviving_cells():
    """The grid tilt must be measured, not asserted."""
    cells = tuple(
        [_cell(5, True) for _ in range(4)] + [_cell(60, False) for _ in range(5)]
    )
    text = diagnose(
        make_scan(pass_rate=4 / 9, n_cells=9, cells=cells),
        run_gates(make_scan(pass_rate=4 / 9, n_cells=9, cells=cells)),
        "REFINE",
    )
    assert "tilts short" in text
    assert "passing-cell mean 5" in text


@pytest.mark.parametrize("winners,losers", [
    # means 33.3 vs 46.7 — a 1.4x gap, below the 1.5x bar, so stay quiet
    ([20, 20, 60], [20, 60, 60]),
    # means identical — nothing to say
    ([5, 60], [5, 60]),
])
def test_diagnosis_does_not_invent_a_tilt_from_noise(winners, losers):
    """The grid is {5, 20, 60}; a sub-1.5x gap is sampling noise, not a lead."""
    cells = tuple([_cell(w, True) for w in winners] + [_cell(w, False) for w in losers])
    s = make_scan(pass_rate=len(winners) / len(cells), n_cells=len(cells), cells=cells)
    assert "no clear window tilt" in diagnose(s, run_gates(s), "REFINE")


# ======================================================================
# 3. GATE_5 catches redundancy
# ======================================================================


def test_near_copy_of_an_accepted_factor_is_rejected():
    result = judge(make_scan(), max_ic_vs_sota=0.95)
    gates = {g.name: g for g in result.gates}

    assert not gates["GATE_5"].passed
    assert result.verdict == "REJECT", "a near-duplicate will stay one under mutation"
    assert result.reject_reason == "redundant_with_sota"
    assert any("Switch theme" in s for s in result.suggestions)


def test_gate_5_passes_on_an_empty_sota_set():
    """The first accepted factor cannot be redundant with anything."""
    gates = {g.name: g for g in run_gates(make_scan(), max_ic_vs_sota=0.0)}
    assert gates["GATE_5"].passed


def test_redundancy_outranks_every_other_failure():
    """Even a structure failing four gates is REJECTed for redundancy first."""
    result = judge(
        make_scan(sharpe_train=0.9, sharpe_val=0.05, pass_rate=0.1, turnover=99.0),
        max_ic_vs_sota=0.99,
    )
    assert result.verdict == "REJECT"
    assert result.reject_reason == "redundant_with_sota"


# ======================================================================
# 4. Failure write-back and actionable diagnosis
# ======================================================================


def test_reject_is_written_to_knowledge_failures(tmp_path):
    s = make_scan()
    result = judge(s, max_ic_vs_sota=0.95)
    path = record_failure(result, s, directory=tmp_path)

    assert path is not None and path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Disproved" in text
    assert s.spec_hash[:12] in text
    assert s.structure_label in text
    assert "GATE_5" in text


def test_refine_is_not_written_to_failures(tmp_path):
    """REFINE is a live branch, not a disproved path."""
    s = make_scan(sharpe_train=0.9, sharpe_val=0.1)
    result = judge(s)
    assert result.verdict == "REFINE"
    assert record_failure(result, s, directory=tmp_path) is None
    assert not list(tmp_path.glob("*.md"))


def test_repeated_rejects_append_to_one_file(tmp_path):
    s = make_scan()
    result = judge(s, max_ic_vs_sota=0.95)
    record_failure(result, s, directory=tmp_path)
    record_failure(result, s, directory=tmp_path)
    assert len(list(tmp_path.glob("*.md"))) == 1
    assert tmp_path.joinpath("redundant_with_sota.md").read_text(
        encoding="utf-8").count("## `") == 2


@pytest.mark.parametrize("kwargs,must_contain", [
    (dict(sharpe_train=0.9, sharpe_val=0.1), "Reduce complexity"),
    (dict(pass_rate=0.1), "rather than the window"),
    (dict(turnover=99.0), "slower operator"),
    (dict(nan_ratio=0.8), "monthly"),
])
def test_every_failure_mode_yields_an_executable_instruction(kwargs, must_contain):
    '''"Sharpe too low" is not a suggestion. Each string must name a spec field.'''
    result = judge(make_scan(**kwargs))
    joined = " ".join(result.suggestions)
    assert must_contain in joined, f"{kwargs} -> {joined}"
    assert joined, "a failing gate with no instruction wastes the trial"


def test_diagnosis_never_tells_the_agent_to_tune_windows():
    """A4: windows are not in the spec, so 'try other windows' is unactionable."""
    for kwargs in [dict(pass_rate=0.1), dict(sharpe_train=0.9, sharpe_val=0.1),
                   dict(turnover=99.0), dict(nan_ratio=0.8), dict(sharpe_val=-0.3)]:
        low = judge(make_scan(**kwargs)).diagnosis.lower()
        for banned in ("tune the window", "adjust the window",
                       "try other window", "other windows", "different windows"):
            assert banned not in low, (banned, low)


# ======================================================================
# C1 / C5 — structural guarantees
# ======================================================================


def test_analyst_calls_no_model():
    """C1: the gate layer must contain no LLM client, by parse not by promise."""
    tree = ast.parse((config.PROJECT_ROOT / "harness" / "analyst.py").read_text(
        encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"openai", "anthropic", "httpx", "requests", "urllib", "litellm"}
    assert not (imported & banned), f"analyst.py imports {imported & banned}"


@pytest.mark.parametrize("bump", [0.0, 1.0, 10.0, 1000.0])
def test_sharpe_max_cannot_change_the_verdict(bump):
    """C5 as an experiment: inflate the peak, demand an identical judgement."""
    base = make_scan(sharpe_train=0.40, sharpe_val=0.10, pass_rate=0.2)
    tampered = dataclasses.replace(base, sharpe_max_val=base.sharpe_max_val + bump)

    a, b = judge(base), judge(tampered)
    assert a.verdict == b.verdict
    assert a.failed_gates == b.failed_gates
    assert a.fitness == b.fitness
    assert a.suggestions == b.suggestions


def test_run_gates_never_reads_sharpe_max():
    """The same claim at source level, so a future edit cannot sneak it in."""
    source = (config.PROJECT_ROOT / "harness" / "analyst.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("run_gates", "decide"):
            reads = [
                n.attr for n in ast.walk(node)
                if isinstance(n, ast.Attribute) and "sharpe_max" in n.attr
            ]
            assert not reads, f"{node.name} reads {reads} (CLAUDE.md C5)"


def test_fitness_is_the_conservative_estimate():
    s = make_scan(sharpe_train=0.80, sharpe_val=0.20, sharpe_max=3.0)
    assert judge(s).fitness == pytest.approx(0.20)
    s2 = make_scan(sharpe_train=0.10, sharpe_val=0.90, sharpe_max=3.0)
    assert judge(s2).fitness == pytest.approx(0.10)


# ======================================================================
# Three-state logic
# ======================================================================


def test_all_gates_passing_is_accept():
    assert judge(make_scan()).verdict == "ACCEPT"
    assert judge(make_scan()).reject_reason == ""


def test_dead_everywhere_is_reject_not_refine():
    s = make_scan(sharpe_train=0.3, sharpe_val=-0.4, pass_rate=0.0)
    result = judge(s)
    assert result.verdict == "REJECT"
    assert result.reject_reason == "no_signal_in_any_grid_cell"


def test_one_working_cell_still_earns_a_refine():
    """pass_rate > 0 means some corner of the structure holds."""
    s = make_scan(sharpe_train=0.3, sharpe_val=-0.4, pass_rate=1 / 27, n_cells=27)
    assert judge(s).verdict == "REFINE"


def test_a_high_peak_alone_does_not_earn_a_refine():
    """C5 again, at the decision boundary rather than inside a gate."""
    dead = make_scan(sharpe_train=0.3, sharpe_val=-0.4, pass_rate=0.0, sharpe_max=5.0)
    assert judge(dead).verdict == "REJECT"


def test_verdicts_are_known_to_the_journal():
    from harness.journal import VERDICTS

    for kwargs in [dict(), dict(pass_rate=0.0, sharpe_val=-0.1),
                   dict(sharpe_train=0.9, sharpe_val=0.1)]:
        assert judge(make_scan(**kwargs)).verdict in VERDICTS


def test_judge_is_deterministic():
    s = make_scan(sharpe_train=0.5, sharpe_val=0.2, pass_rate=0.3)
    assert judge(s).as_dict() == judge(s).as_dict()


# ======================================================================
# C7 — the journal round-trip
# ======================================================================


def test_verdict_survives_a_journal_round_trip(tmp_path):
    spec = L0_SPECS["pmf_051056"]
    s = make_scan(sharpe_train=0.9, sharpe_val=0.1, n_cells=spec.n_grid_cells)
    result = judge(s)

    path = tmp_path / "trials.jsonl"
    append_trial(
        TrialRecord.from_spec(
            spec, trial_id=1, verdict=result.verdict, stage="gate",
            source="l0", **journal_fields(result, s),
        ),
        path=path,
    )

    (back,) = read_trials(path)
    assert back.verdict == "REFINE"
    assert back.diagnosis == result.diagnosis
    assert back.gates["GATE_2"]["passed"] is False
    assert back.metrics["fitness"] == pytest.approx(s.fitness)
    assert back.metrics["sharpe_max_val"] == pytest.approx(s.sharpe_max_val), (
        "sharpe_max must be recorded even though it decides nothing (C5)"
    )
    assert back.n_grid_cells == s.n_cells, (
        "the spec's own grid size and the number of cells actually scanned "
        "must agree; journal_fields omits n_grid_cells so from_spec owns it"
    )


# ======================================================================
# 5. The L0 gate table — real data, real thresholds
# ======================================================================


@pytest.fixture(scope="module")
def l0_judgements():
    inds = load_indicators()
    returns = load_returns()
    out = {}
    for name, spec in L0_SPECS.items():
        s = scan(spec, inds, returns)
        out[name] = (s, judge(s))
    return out


def test_l0_gate_table(l0_judgements, capsys):
    """Print the calibration table plan.md asks for, and sanity-check it."""
    lines = ["", f"{'factor':<12}{'verdict':<9}{'fit':>7}  " + " ".join(GATE_ORDER)]
    counts = {g: 0 for g in GATE_ORDER}
    verdicts = {}
    for name in sorted(l0_judgements):
        s, v = l0_judgements[name]
        marks = []
        for g in v.gates:
            counts[g.name] += int(g.passed)
            marks.append(" o     " if g.passed else " X     ")
        verdicts[v.verdict] = verdicts.get(v.verdict, 0) + 1
        lines.append(f"{name:<12}{v.verdict:<9}{s.fitness:>7.3f}  " + "".join(marks))
    lines.append("")
    lines.append("pass counts: " + ", ".join(f"{g} {counts[g]}/15" for g in GATE_ORDER))
    lines.append("verdicts: " + str(verdicts))
    with capsys.disabled():
        print("\n".join(lines))

    # The thresholds must neither wave everything through nor kill everything.
    assert verdicts.get("ACCEPT", 0) >= 1, "no L0 factor survives — gates are too tight"
    assert verdicts.get("ACCEPT", 0) <= 10, "almost everything passes — gates are inert"
    assert counts["GATE_4"] == 15, (
        "turnover cap is currently non-binding on L0; if this changes, "
        "revisit BACKLOG B6 before trusting GATE_4"
    )


def test_every_l0_factor_gets_a_diagnosis(l0_judgements):
    for name, (s, v) in l0_judgements.items():
        assert v.diagnosis, name
        assert str(s.n_cells) in v.diagnosis
        assert "sharpe_max" in v.diagnosis
        if v.verdict != "ACCEPT":
            assert v.suggestions, f"{name} failed {v.failed_gates} with no instruction"
