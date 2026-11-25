"""⑤ Analysis Unit — six gates, a fitness score, and a repair instruction.

The gates are plain `if` statements (C1). No LLM decides anything here. The
diagnosis text is composed deterministically from the grid detail as well: the
project does not need a model to say "20 of 27 cells failed and the survivors
all sit at the short-window end", and a deterministic composer can be tested.
CLAUDE.md permits an LLM to rewrite this prose later; nothing here requires it.

C5 is enforced structurally, not by convention: `sharpe_max_val` is never read
by `run_gates` or `decide`. `test_phase3.py` proves it by tampering with the
value and asserting the verdict does not move.

What the Analysis Unit sees is deliberately narrow (plan.md §2, "local view"):
one scan result plus the IC against the SOTA set. It does not see the trial
history — that is the Synthesis Unit's global view, and mixing the two into one
prompt is the failure mode the split exists to prevent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from harness import config
from harness.validator import CellResult, ScanResult

logger = logging.getLogger(__name__)

GATE_ORDER = ("GATE_1", "GATE_2", "GATE_3", "GATE_4", "GATE_5", "GATE_6")

GATE_TITLES = {
    "GATE_1": "out-of-sample direction",
    "GATE_2": "decay ratio",
    "GATE_3": "window robustness",
    "GATE_4": "turnover",
    "GATE_5": "signal dedup",
    "GATE_6": "data coverage",
}


# ======================================================================
# Gate results
# ======================================================================


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    observed: float
    threshold: float
    note: str

    @property
    def title(self) -> str:
        return GATE_TITLES[self.name]

    def as_dict(self) -> dict:
        return {
            "name": self.name, "passed": self.passed,
            "observed": None if np.isnan(self.observed) else round(float(self.observed), 4),
            "threshold": self.threshold, "note": self.note,
        }


@dataclass(frozen=True)
class Verdict:
    """The Analysis Unit's whole output for one trial."""

    verdict: str                      # ACCEPT | REFINE | REJECT
    fitness: float
    gates: tuple[GateResult, ...]
    failed_gates: tuple[str, ...]
    reject_reason: str
    diagnosis: str
    suggestions: tuple[str, ...]
    spec_hash: str = ""

    @property
    def passed(self) -> bool:
        return self.verdict == "ACCEPT"

    def gate_dict(self) -> dict:
        return {g.name: g.as_dict() for g in self.gates}

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict, "fitness": round(self.fitness, 4),
            "failed_gates": list(self.failed_gates),
            "reject_reason": self.reject_reason,
            "diagnosis": self.diagnosis,
            "suggestions": list(self.suggestions),
            "gates": self.gate_dict(),
        }


# ======================================================================
# The six gates
# ======================================================================


def decay_ratio(sharpe_train: float, sharpe_val: float) -> float:
    """val / train, or NaN when there was nothing in sample to decay from.

    Returning NaN rather than a number is the point: a factor that lost money
    in sample and made money out of sample has no "decay" to measure, and
    handing it a flattering negative-over-negative ratio would let a fluke
    clear an overfitting gate. The same rule runs at cell level in
    `validator._cell_verdict`, so the two layers cannot disagree.
    """
    if sharpe_train <= 0:
        return float("nan")
    return sharpe_val / sharpe_train


def run_gates(scan: ScanResult, max_ic_vs_sota: float = 0.0) -> tuple[GateResult, ...]:
    """Evaluate all six gates. Pure `if`, no LLM (C1), no `sharpe_max` (C5)."""
    s_tr = scan.sharpe_median_train
    s_va = scan.sharpe_median_val
    ratio = decay_ratio(s_tr, s_va)

    g1 = GateResult(
        "GATE_1", s_va > config.GATE_1_MIN_SHARPE_VAL, s_va,
        config.GATE_1_MIN_SHARPE_VAL,
        f"val-segment median Sharpe {s_va:.3f}",
    )

    if np.isnan(ratio):
        g2 = GateResult(
            "GATE_2", False, float("nan"), config.GATE_2_MIN_DECAY_RATIO,
            f"train-segment median Sharpe {s_tr:.3f} ≤ 0, no in-sample performance to decay from",
        )
    else:
        g2 = GateResult(
            "GATE_2", ratio > config.GATE_2_MIN_DECAY_RATIO, ratio,
            config.GATE_2_MIN_DECAY_RATIO,
            f"train {s_tr:.3f} → val {s_va:.3f}, decay ratio {ratio:.2f}",
        )

    n_pass = int(round(scan.pass_rate * scan.n_cells))
    g3 = GateResult(
        "GATE_3", scan.pass_rate > config.GATE_3_MIN_PASS_RATE, scan.pass_rate,
        config.GATE_3_MIN_PASS_RATE,
        f"{scan.n_cells} grid cells, {n_pass} passed ({scan.pass_rate:.0%})",
    )

    turn = scan.ann_turnover_median
    g4 = GateResult(
        "GATE_4", bool(not np.isnan(turn) and turn < config.GATE_4_TURNOVER_CAP),
        turn, config.GATE_4_TURNOVER_CAP,
        f"median annualized turnover {turn:.2f}" if not np.isnan(turn) else "turnover cannot be computed",
    )

    g5 = GateResult(
        "GATE_5", max_ic_vs_sota < config.GATE_5_MAX_IC_VS_SOTA, max_ic_vs_sota,
        config.GATE_5_MAX_IC_VS_SOTA,
        f"max |IC| vs SOTA set = {max_ic_vs_sota:.3f}",
    )

    g6 = GateResult(
        "GATE_6", scan.nan_ratio_median < config.GATE_6_MAX_NAN_RATIO,
        scan.nan_ratio_median, config.GATE_6_MAX_NAN_RATIO,
        f"median share of unusable values in the evaluation window {scan.nan_ratio_median:.1%}",
    )

    return (g1, g2, g3, g4, g5, g6)


# ======================================================================
# Three-state decision
# ======================================================================


def decide(gates: Sequence[GateResult], scan: ScanResult) -> tuple[str, str]:
    """ACCEPT / REFINE / REJECT, plus a one-line machine-readable reason.

    The dividing line between REFINE and REJECT is **whether a mutation of this
    structure could plausibly fix it**, not how badly it scored:

    * REJECT on GATE_5 — a near-duplicate of something already accepted stays a
      near-duplicate under the small mutations the agent makes. Spend the
      budget elsewhere.
    * REJECT when nothing worked anywhere — `pass_rate == 0` and a non-positive
      median in EITHER split. A zero pass_rate already means the structure
      survived no corner of its grid; if on top of that it lost money in sample
      (train <= 0) or out of sample (val <= 0), there is no credible basis to
      mutate it. In particular a train<=0 / val>0 factor is the "lucky on a
      quarter-length val window" case the diagnosis composer already calls
      structurally invalid — keying REJECT off val alone let it leak through as
      REFINE and hand the agent a verdict ("mutate it") that its own diagnosis
      contradicted ("do not mutate it"). Both halves now read the same signal.
      Note this uses pass_rate and the median, never `sharpe_max_val` (C5):
      "one lucky cell" must not buy a second chance.
    * REFINE otherwise — the diagnosis names a specific change to try.
    """
    failed = tuple(g.name for g in gates if not g.passed)
    if not failed:
        return "ACCEPT", ""

    if "GATE_5" in failed:
        return "REJECT", "redundant_with_sota"

    if scan.pass_rate == 0.0 and (
        scan.sharpe_median_val <= 0.0 or scan.sharpe_median_train <= 0.0
    ):
        return "REJECT", "no_signal_in_any_grid_cell"

    return "REFINE", "+".join(failed)


# ======================================================================
# Diagnosis — deterministic, composed from the grid
# ======================================================================


def _window_profile(cells: Sequence[CellResult]) -> str:
    """Where in the grid do the surviving cells sit?

    plan.md's worked example asks the diagnosis to say "concentrated at the
    short-window end", which only means something if it is measured. Compares
    the mean window value of passing versus failing cells, per role.
    """
    winners = [c for c in cells if c.passed]
    losers = [c for c in cells if not c.passed]
    if not winners or not losers:
        return ""

    phrases = []
    for role in config.ROLE_NAMES:
        w = [c.windows[role] for c in winners if role in c.windows]
        l = [c.windows[role] for c in losers if role in c.windows]
        if not w or not l:
            continue
        mw, ml = float(np.mean(w)), float(np.mean(l))
        # Only call it a tilt if the two means differ by at least 1.5x. The
        # grid is {5, 20, 60}, so 32 vs 27 is one cell of sampling noise, and
        # reporting it as "tilts long" would hand the agent a false lead.
        lo, hi = min(mw, ml), max(mw, ml)
        if lo <= 0 or hi / lo < 1.5:
            continue
        side = "short" if mw < ml else "long"
        phrases.append(f"{role} tilts {side} (passing-cell mean {mw:.0f} vs failing {ml:.0f})")

    if not phrases:
        return "the passing cells show no clear window tilt in the grid"
    return "passing cells concentrate at: " + "; ".join(phrases)


def _suggestions(gates: Sequence[GateResult], scan: ScanResult) -> tuple[str, ...]:
    """One concrete, executable instruction per failed gate.

    "Sharpe too low" is not actionable. "Swap the direction leg's ma_cross for
    ma_diff" is. Every string here names a field of the spec the agent can
    actually change — it never says "tune the windows", because the validator
    already scanned every window (A4).
    """
    failed = {g.name: g for g in gates if not g.passed}
    out: list[str] = []

    if "GATE_6" in failed:
        out.append(
            "First fix data coverage: one input is likely a monthly indicator "
            "(PMF0018 / PMF0031 / PMF0036 / PMF0042) bound to a short-window role; see R1. "
            "Swap it for a daily series, or use an operator that does not divide by the standard deviation."
        )
    if "GATE_5" in failed:
        out.append(
            f"Redundant with the SOTA set (|IC| {failed['GATE_5'].observed:.2f}). "
            "Switch theme, or at least swap one input indicator — tuning operators within the same theme will most likely stay redundant."
        )
    if "GATE_2" in failed:
        if np.isnan(failed["GATE_2"].observed):
            out.append(
                "The train segment itself does not make money; this is not overfitting but an invalid structure. "
                "Swap the operator or the indicator pair; do not keep mutating this structure."
            )
            # The structure is not overfit, so do not also tell it to simplify.
        else:
            out.append(
                f"Decay ratio {failed['GATE_2'].observed:.2f}, classic overfitting. Reduce complexity: "
                "drop one pre-transform, or switch the two-leg template to identity keeping only the main signal."
            )
    if "GATE_3" in failed and "GATE_2" not in failed:
        out.append(
            "The structure holds in only a few windows — a parameter accident. Change the operator rather than the window — "
            "the window grid has already been fully swept, there are no unsearched parameters."
        )
    if "GATE_4" in failed:
        out.append(
            f"Turnover {failed['GATE_4'].observed:.1f} exceeds the cap {config.GATE_4_TURNOVER_CAP:.0f}. "
            "Switch the direction leg to a slower operator (ma_cross → ma_diff), "
            "or add a smoothing layer to the intensity leg."
        )
    if "GATE_1" in failed and "GATE_2" not in failed and "GATE_3" not in failed:
        out.append(
            "The val-segment direction is negative while window robustness is acceptable, suggesting the sign may be "
            "skewed by a particular regime in the train segment. Check whether the hypothesis's economic direction is inverted."
        )
    return tuple(out)


def diagnose(scan: ScanResult, gates: Sequence[GateResult], verdict: str) -> str:
    """Compose the diagnosis paragraph the next round actually reads."""
    n_pass = int(round(scan.pass_rate * scan.n_cells))
    parts = [f"{scan.n_cells} grid cells, {n_pass} passed (pass_rate {scan.pass_rate:.2f})"]

    profile = _window_profile(scan.grid_detail)
    if profile:
        parts.append(profile)

    ratio = decay_ratio(scan.sharpe_median_train, scan.sharpe_median_val)
    if np.isnan(ratio):
        parts.append(
            f"train {scan.sharpe_median_train:.3f} / val {scan.sharpe_median_val:.3f}, "
            f"train non-positive, decay ratio undefined"
        )
    else:
        parts.append(
            f"train {scan.sharpe_median_train:.3f} / val {scan.sharpe_median_val:.3f}, "
            f"decay ratio {ratio:.2f}"
        )
    parts.append(f"fitness {scan.fitness:.3f}")

    head = "; ".join(parts) + ". "

    failed = [g for g in gates if not g.passed]
    if not failed:
        body = f"All six gates passed; added to the SOTA set. Median turnover {scan.ann_turnover_median:.1f}. "
    else:
        names = ", ".join(f"{g.name}({g.title})" for g in failed)
        body = f"Failed: {names}. " + " ".join(_suggestions(gates, scan))

    # C5, stated where a future reader will see it rather than buried in a doc.
    tail = f" (sharpe_max {scan.sharpe_max_val:.3f} recorded only, not part of the verdict)"
    return f"{head}{body}{tail}"


# ======================================================================
# Top level
# ======================================================================


def judge(scan: ScanResult, max_ic_vs_sota: float = 0.0) -> Verdict:
    """Scan result -> verdict + diagnosis. The whole Analysis Unit in one call."""
    gates = run_gates(scan, max_ic_vs_sota)
    verdict, reason = decide(gates, scan)
    return Verdict(
        verdict=verdict,
        fitness=scan.fitness,
        gates=gates,
        failed_gates=tuple(g.name for g in gates if not g.passed),
        reject_reason=reason,
        diagnosis=diagnose(scan, gates, verdict),
        suggestions=_suggestions(gates, scan),
        spec_hash=scan.spec_hash,
    )


# ======================================================================
# Failure write-back (plan.md Phase 3 item 3)
# ======================================================================


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower()).strip("_") or "unknown"


def record_failure(verdict: Verdict, scan: ScanResult,
                   directory=None) -> "object | None":
    """Append a REJECT to `knowledge/failures/`, grouped by reason.

    The third bucket of the knowledge base: `rules` are verified constraints,
    `findings` are observations, `failures` are paths already disproved. Only
    REJECT is written — a REFINE is still a live branch and lives in the
    journal's diagnosis field, not in the "do not go here again" file.
    """
    if verdict.verdict != "REJECT":
        return None

    directory = directory or config.FAILURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{_slug(verdict.reject_reason)}.md"

    if not path.exists():
        path.write_text(
            f"# Disproved: {verdict.reject_reason}\n\n"
            f"> Written automatically by `harness/analyst.py`. Each entry is a REJECTed "
            f"structure; do not retry it.\n",
            encoding="utf-8",
        )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n## `{scan.spec_hash[:12]}`\n\n"
            f"- Structure: `{scan.structure_label}`\n"
            f"- Failed gates: {', '.join(verdict.failed_gates)}\n"
            f"- Diagnosis: {verdict.diagnosis}\n"
        )
    return path


def journal_fields(verdict: Verdict, scan: ScanResult) -> dict:
    """The metrics / gates / diagnosis payload for a `TrialRecord` (C7).

    `n_grid_cells` is deliberately absent: `TrialRecord.from_spec` already
    derives it from the spec, and returning it here would collide. The two must
    agree, which `test_phase3.py` checks rather than assumes.
    """
    return {
        "metrics": {
            "fitness": round(scan.fitness, 4),
            "sharpe_median_train": round(scan.sharpe_median_train, 4),
            "sharpe_median_val": round(scan.sharpe_median_val, 4),
            "sharpe_iqr_val": round(scan.sharpe_iqr_val, 4),
            "pass_rate": round(scan.pass_rate, 4),
            "sharpe_max_val": round(scan.sharpe_max_val, 4),
            "ann_turnover_median": round(scan.ann_turnover_median, 4),
            "nan_ratio_median": round(scan.nan_ratio_median, 4),
        },
        "gates": verdict.gate_dict(),
        "diagnosis": verdict.diagnosis,
        "reject_reason": verdict.reject_reason,
    }


__all__ = [
    "GATE_ORDER",
    "GATE_TITLES",
    "GateResult",
    "Verdict",
    "decay_ratio",
    "decide",
    "diagnose",
    "journal_fields",
    "judge",
    "record_failure",
    "run_gates",
]
