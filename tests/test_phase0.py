"""Phase 0 acceptance — Specification Unit.

plan.md §5 Phase 0 asks for:
  * same spec hashes identically; shuffling field order does not change the hash
  * an illegal operator / off-whitelist indicator raises with a human reason
  * journal round-trips, and check_if_tried hits

Plus one test the plan does not ask for but Phase 1's kill criterion depends on:
all 15 L0 factors must be expressible in the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import config
from harness.l0_catalog import L0_SPECS, L0_SURVIVORS, L0_WINDOWS
from harness.spec import (
    INDICATORS,
    OPERATORS,
    TEMPLATES,
    THEMES,
    FactorSpec,
    SpecError,
    SpecInput,
    SpecLeg,
    window_grid,
)
from harness import journal


# ======================================================================
# Fixtures
# ======================================================================


def simple_spec(**overrides) -> FactorSpec:
    base = dict(
        theme="risk",
        template="direction_x_intensity",
        legs={
            "direction": SpecLeg("ma_diff", (SpecInput("PMF0007"),), ("w1", "w1")),
            "intensity": SpecLeg("abs_pct_change", (SpecInput("PMF0042"),), ("w2",)),
        },
        hypothesis="Geopolitical escalation plus an inflation shock lifts safe-haven demand.",
    )
    base.update(overrides)
    return FactorSpec(**base)


@pytest.fixture
def tmp_journal(tmp_path: Path) -> Path:
    return tmp_path / "trials.jsonl"


# ======================================================================
# Hash
# ======================================================================


def test_same_spec_hashes_identically():
    assert simple_spec().spec_hash == simple_spec().spec_hash


def test_hash_is_stable_under_key_order():
    """plan.md: the hash stays identical when field order is shuffled."""
    a = {
        "theme": "risk",
        "template": "direction_x_intensity",
        "hypothesis": "h",
        "legs": {
            "direction": {"op": "ma_diff", "inputs": [{"indicator": "PMF0007"}],
                          "op_windows": ["w1", "w1"]},
            "intensity": {"op": "abs_pct_change", "inputs": [{"indicator": "PMF0042"}],
                          "op_windows": ["w2"]},
        },
    }
    b = {
        "legs": {
            "intensity": {"inputs": [{"indicator": "PMF0042"}], "op": "abs_pct_change",
                          "op_windows": ["w2"]},
            "direction": {"inputs": [{"indicator": "PMF0007"}], "op": "ma_diff",
                          "op_windows": ["w1", "w1"]},
        },
        "hypothesis": "h",
        "template": "direction_x_intensity",
        "theme": "risk",
    }
    assert FactorSpec.from_dict(a).spec_hash == FactorSpec.from_dict(b).spec_hash


def test_hash_ignores_narrative_but_tracks_structure():
    """Retelling the economics must not dodge dedup; swapping an indicator must."""
    same_structure = simple_spec(
        hypothesis="A completely different story about the same computation.",
        mutation_note="re-pitched",
        parent_spec_hash="deadbeef",
    )
    assert same_structure.spec_hash == simple_spec().spec_hash

    changed = simple_spec(
        legs={
            "direction": SpecLeg("ma_diff", (SpecInput("PMF0018"),), ("w1", "w1")),
            "intensity": SpecLeg("abs_pct_change", (SpecInput("PMF0042"),), ("w2",)),
        }
    )
    assert changed.spec_hash != simple_spec().spec_hash


def test_hash_distinguishes_input_order():
    """ratio_over_logsize(x, y) is not ratio_over_logsize(y, x)."""
    xy = FactorSpec(
        theme="etf", template="direction_x_intensity",
        legs={
            "direction": SpecLeg("ma_diff", (SpecInput("PMF0031"),), ("w1", "w1")),
            "intensity": SpecLeg("ratio_over_logsize",
                                 (SpecInput("PMF0013", ("abs",)), SpecInput("PMF0031")),
                                 ("w1",)),
        },
        hypothesis="h",
    )
    yx = FactorSpec(
        theme="etf", template="direction_x_intensity",
        legs={
            "direction": SpecLeg("ma_diff", (SpecInput("PMF0031"),), ("w1", "w1")),
            "intensity": SpecLeg("ratio_over_logsize",
                                 (SpecInput("PMF0031"), SpecInput("PMF0013", ("abs",))),
                                 ("w1",)),
        },
        hypothesis="h",
    )
    assert xy.spec_hash != yx.spec_hash


def test_json_round_trip_preserves_hash():
    spec = simple_spec()
    revived = FactorSpec.from_json(json.dumps(spec.to_dict()))
    assert revived.spec_hash == spec.spec_hash
    assert revived.to_dict() == spec.to_dict()


# ======================================================================
# Validation — every error must name the problem in plain language
# ======================================================================


def test_unknown_operator_is_rejected_with_a_reason():
    with pytest.raises(SpecError) as exc:
        simple_spec(legs={
            "direction": SpecLeg("ma_diffff", (SpecInput("PMF0007"),), ("w1", "w1")),
            "intensity": SpecLeg("abs_pct_change", (SpecInput("PMF0042"),), ("w2",)),
        })
    msg = str(exc.value)
    assert "ma_diffff" in msg
    assert "Allowed operators" in msg
    assert "ma_diff" in msg          # the fix is spelled out, not just the complaint


def test_off_whitelist_indicator_is_rejected_with_a_reason():
    with pytest.raises(SpecError) as exc:
        simple_spec(legs={
            "direction": SpecLeg("ma_diff", (SpecInput("PMF9999"),), ("w1", "w1")),
            "intensity": SpecLeg("abs_pct_change", (SpecInput("PMF0042"),), ("w2",)),
        })
    msg = str(exc.value)
    assert "PMF9999" in msg and "Allowed indicators" in msg


def test_unknown_theme_and_template_are_rejected():
    with pytest.raises(SpecError, match="unknown theme"):
        simple_spec(theme="crypto")
    with pytest.raises(SpecError, match="unknown template"):
        simple_spec(template="direction_x_vibes")


def test_empty_hypothesis_is_rejected():
    """Hypothesis-first is a schema rule, not an honour system."""
    with pytest.raises(SpecError, match="hypothesis is empty"):
        simple_spec(hypothesis="   ")


def test_wrong_operator_kind_for_the_role_is_rejected():
    """An intensity operator cannot fill the direction role."""
    with pytest.raises(SpecError) as exc:
        simple_spec(legs={
            "direction": SpecLeg("abs_zscore", (SpecInput("PMF0007"),), ("w1",)),
            "intensity": SpecLeg("abs_pct_change", (SpecInput("PMF0042"),), ("w2",)),
        })
    msg = str(exc.value)
    assert "produces" in msg and "Operators that fit here" in msg


def test_missing_leg_is_rejected():
    with pytest.raises(SpecError, match="Missing: intensity"):
        simple_spec(legs={
            "direction": SpecLeg("ma_diff", (SpecInput("PMF0007"),), ("w1", "w1")),
        })


def test_wrong_arity_is_rejected():
    with pytest.raises(SpecError, match="takes 2 indicator"):
        FactorSpec(
            theme="rates", template="identity",
            legs={"value": SpecLeg("corr_change", (SpecInput("PMF0002", ("diff",)),),
                                   ("w1", "w1"))},
            hypothesis="h",
        )


def test_too_many_pre_transforms_is_rejected():
    with pytest.raises(SpecError, match="complexity control"):
        simple_spec(legs={
            "direction": SpecLeg("ma_diff",
                                 (SpecInput("PMF0007", ("diff", "abs", "lag1")),),
                                 ("w1", "w1")),
            "intensity": SpecLeg("abs_pct_change", (SpecInput("PMF0042"),), ("w2",)),
        })


def test_degenerate_ordered_slots_are_rejected():
    """ma_cross with both windows on one role is identically zero."""
    with pytest.raises(SpecError, match="identically zero"):
        simple_spec(legs={
            "direction": SpecLeg("ma_cross", (SpecInput("PMF0007"),), ("w1", "w1")),
            "intensity": SpecLeg("abs_pct_change", (SpecInput("PMF0042"),), ("w2",)),
        })


# ======================================================================
# Window grid — A4
# ======================================================================


def test_no_window_value_appears_in_a_serialised_spec():
    """A4: a spec names roles, never the numbers 5 / 20 / 60."""
    blob = json.dumps(simple_spec().to_dict())
    for value in config.WINDOW_VALUES:
        assert f": {value}" not in blob and f'"{value}"' not in blob


def test_grid_sizes_follow_the_number_of_roles():
    one_role = FactorSpec(
        theme="etf", template="signal_x_persistence",
        legs={
            "signal": SpecLeg("aroon_balance", (SpecInput("PMF0031", ("pct_ret",)),), ("w1",)),
            "persistence": SpecLeg("path_efficiency",
                                   (SpecInput("PMF0031", ("pct_ret",)),), ("w1",)),
        },
        hypothesis="h",
    )
    assert one_role.window_roles == ("w1",)
    assert one_role.n_grid_cells == len(config.WINDOW_VALUES)        # 3

    two_roles = simple_spec()
    assert two_roles.window_roles == ("w1", "w2")
    assert two_roles.n_grid_cells == len(config.WINDOW_VALUES) ** 2  # 9


def test_grid_never_exceeds_the_cap():
    for name, spec in L0_SPECS.items():
        assert 0 < spec.n_grid_cells <= config.MAX_GRID_CELLS, name


def test_ordered_slot_cells_are_filtered_out():
    """Every ma_cross cell in the grid must have a strictly longer slow window."""
    spec = L0_SPECS["pmf_069545"]
    cells = window_grid(spec)
    assert cells
    for cell in cells:
        assert cell["w3"] > cell["w2"]


# ======================================================================
# L0 coverage — Phase 1's kill criterion, checked early
# ======================================================================


def test_all_15_l0_factors_are_expressible():
    assert len(L0_SPECS) == 15, "L0 is defined as 15 hand-designed factors"
    for name, spec in L0_SPECS.items():
        assert isinstance(spec, FactorSpec), name
        assert spec.spec_hash, name


def test_all_15_l0_structures_are_distinct():
    hashes = {name: spec.spec_hash for name, spec in L0_SPECS.items()}
    assert len(set(hashes.values())) == 15, f"collision among L0 structures: {hashes}"


def test_l0_window_bindings_cover_every_role_and_sit_inside_the_grid():
    for name, spec in L0_SPECS.items():
        bound = L0_WINDOWS[name]
        assert set(bound) == set(spec.window_roles), (
            f"{name}: recorded windows {sorted(bound)} != roles {list(spec.window_roles)}"
        )
        for value in bound.values():
            assert value in config.WINDOW_VALUES, f"{name}: {value} not in the grid"
        assert bound in window_grid(spec), (
            f"{name}: the original window choice is not a valid grid cell"
        )


def test_l0_spans_all_five_themes_and_all_five_templates():
    assert {s.theme for s in L0_SPECS.values()} == set(THEMES)
    assert {s.template for s in L0_SPECS.values()} == set(TEMPLATES)


def test_l0_survivors_are_a_subset_of_l0():
    assert len(L0_SURVIVORS) == 6
    assert set(L0_SURVIVORS) <= set(L0_SPECS)


def test_every_whitelisted_operator_is_reachable_by_some_template():
    """No dead operators: each one can legally fill at least one role."""
    reachable = {
        name for name, op in OPERATORS.items()
        if any(op.kind in kinds for roles in TEMPLATES.values() for kinds in roles.values())
    }
    assert reachable == set(OPERATORS), f"unreachable: {set(OPERATORS) - reachable}"


def test_every_indicator_has_a_valid_theme():
    for code, ind in INDICATORS.items():
        assert ind.theme in THEMES, code


# ======================================================================
# Journal — C7
# ======================================================================


def test_journal_round_trips(tmp_journal: Path):
    spec = simple_spec()
    record = journal.TrialRecord.from_spec(
        spec, trial_id=1, verdict="REJECT", stage="gate",
        reject_reason="GATE_3", metrics={"sharpe_median_val": 0.1, "pass_rate": 0.2},
        diagnosis="2 of 9 grid cells pass; structure is fragile, change the operator.",
    )
    journal.append_trial(record, tmp_journal)

    back = journal.read_trials(tmp_journal)
    assert len(back) == 1
    assert back[0].spec_hash == spec.spec_hash
    assert back[0].metrics["pass_rate"] == 0.2
    assert back[0].theme == "risk"
    assert back[0].timestamp


def test_check_if_tried_hits_and_misses(tmp_journal: Path):
    spec = simple_spec()
    assert journal.check_if_tried(spec, tmp_journal) is None

    journal.append_trial(
        journal.TrialRecord.from_spec(spec, trial_id=1, verdict="REJECT", stage="gate"),
        tmp_journal,
    )
    hit = journal.check_if_tried(spec, tmp_journal)
    assert hit is not None and hit.verdict == "REJECT"

    # A re-pitch with new narrative but identical structure must still hit.
    assert journal.check_if_tried(
        simple_spec(hypothesis="totally new story"), tmp_journal
    ) is not None


def test_failed_trials_are_recorded_too(tmp_journal: Path):
    """C7: validation failures and dedup hits get rows, not silence."""
    for i, (verdict, stage) in enumerate(
        [("INVALID", "validate"), ("DUPLICATE", "dedup_struct"),
         ("REDUNDANT", "backtest"), ("ACCEPT", "gate")], start=1
    ):
        journal.append_trial(
            journal.TrialRecord(trial_id=i, spec_hash=f"h{i}", verdict=verdict,
                                stage=stage, theme="risk"),
            tmp_journal,
        )
    s = journal.summary(tmp_journal)
    assert s["n_trials"] == 4
    assert s["n_accepted"] == 1
    assert s["verdicts"]["INVALID"] == 1
    assert journal.next_trial_id(tmp_journal) == 5


def test_unknown_verdict_is_rejected():
    with pytest.raises(ValueError, match="unknown verdict"):
        journal.TrialRecord(trial_id=1, spec_hash="x", verdict="MAYBE", stage="gate")


def test_sota_set_is_sorted_by_fitness_and_sliceable_by_theme(tmp_journal: Path):
    rows = [
        ("risk", "ACCEPT", 0.30), ("risk", "ACCEPT", 0.55),
        ("rates", "ACCEPT", 0.41), ("risk", "REJECT", 0.90),
    ]
    for i, (theme, verdict, fitness) in enumerate(rows, start=1):
        journal.append_trial(
            journal.TrialRecord(trial_id=i, spec_hash=f"h{i}", verdict=verdict,
                                stage="gate", theme=theme,
                                metrics={"fitness": fitness}),
            tmp_journal,
        )
    allsota = journal.get_sota_set(path=tmp_journal)
    assert [r.metrics["fitness"] for r in allsota] == [0.55, 0.41, 0.30]
    risk_only = journal.get_sota_set(theme="risk", path=tmp_journal)
    assert [r.metrics["fitness"] for r in risk_only] == [0.55, 0.30]


def test_recent_diagnoses_returns_newest_first(tmp_journal: Path):
    for i in range(1, 6):
        journal.append_trial(
            journal.TrialRecord(trial_id=i, spec_hash=f"h{i}", verdict="REFINE",
                                stage="gate", theme="risk",
                                diagnosis=f"diagnosis {i}"),
            tmp_journal,
        )
    recent = journal.recent_diagnoses(3, path=tmp_journal)
    assert [r.diagnosis for r in recent] == ["diagnosis 5", "diagnosis 4", "diagnosis 3"]


def test_failures_by_reason_histogram(tmp_journal: Path):
    for i, reason in enumerate(["GATE_3", "GATE_3", "GATE_2"], start=1):
        journal.append_trial(
            journal.TrialRecord(trial_id=i, spec_hash=f"h{i}", verdict="REJECT",
                                stage="gate", reject_reason=reason),
            tmp_journal,
        )
    assert journal.failures_by_reason(path=tmp_journal) == {"GATE_3": 2, "GATE_2": 1}


def test_missing_journal_reads_as_empty(tmp_path: Path):
    assert journal.read_trials(tmp_path / "nope.jsonl") == []
    assert journal.next_trial_id(tmp_path / "nope.jsonl") == 1


# ======================================================================
# C2 — the test split must not be reachable from ordinary config
# ======================================================================


def test_splits_mapping_excludes_test():
    assert set(config.SPLITS) == {"train", "val"}
    assert config.ALLOWED_SPLITS == ("train", "val")


def test_splits_do_not_overlap_and_respect_the_data_cutoff():
    assert config.TRAIN_SPLIT[1] < config.VAL_SPLIT[0]
    assert config.VAL_SPLIT[1] < config.TEST_SPLIT[0]
    assert config.TEST_SPLIT[1] == config.DATA_CUTOFF


def test_every_indicator_covers_the_whole_evaluation_window():
    """Phase -1 kill criterion: enough indicators survive into the test window."""
    usable = [c for c, i in INDICATORS.items() if i.available_through >= config.TEST_SPLIT[0]]
    assert len(usable) >= 5, f"only {len(usable)} indicators reach the test window"
    assert len(usable) == len(INDICATORS)
