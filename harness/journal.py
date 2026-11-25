"""Memory — append-only trial log with structured queries.

CLAUDE.md C7: *every* trial is written, including schema-validation failures,
dedup hits and gate rejections. Failures are the next round's input.

A2: no embeddings, no vector store. Recall is a structured query over JSONL.
Per the patterns note §B, queries can be sliced by theme, because the agent
asks "what have I tried in `risk`", not "what have I tried ever".
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from harness import config
from harness.spec import FactorSpec

logger = logging.getLogger(__name__)


# Terminal states a trial can end in. Anything that is not ACCEPT still gets a
# row — that is the whole point of C7.
VERDICTS = (
    "ACCEPT",      # cleared all six gates, joins the SOTA set
    "REFINE",      # near miss, carries a repair-instruction diagnosis
    "REJECT",      # failed a gate outright
    "DUPLICATE",   # spec hash already tried (structural dedup, pre-backtest)
    "REDUNDANT",   # signal dedup: IC vs SOTA over threshold (pre-gate)
    "INVALID",     # failed schema validation, never reached the builder
    "MISALIGNED",  # hypothesis did not match the structure (LLM-as-judge)
    "ERROR",       # the deterministic engine raised
)

# Which stage the trial stopped at. Lets the report show where budget was spent.
STAGES = ("validate", "dedup_struct", "complexity", "align", "backtest", "gate")


@dataclass
class TrialRecord:
    """One row of journal/trials.jsonl."""

    trial_id: int
    spec_hash: str
    verdict: str
    stage: str
    theme: str = ""
    template: str = ""
    hypothesis: str = ""
    source: str = "agent"              # agent | random | exhaustive | l0
    spec: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    diagnosis: str = ""
    reject_reason: str = ""
    parent_spec_hash: str | None = None
    n_grid_cells: int = 0
    complexity: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(
                f'unknown verdict "{self.verdict}". Allowed: {", ".join(VERDICTS)}'
            )
        if self.stage not in STAGES:
            raise ValueError(
                f'unknown stage "{self.stage}". Allowed: {", ".join(STAGES)}'
            )
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_json(line: str) -> "TrialRecord":
        return TrialRecord(**json.loads(line))

    @staticmethod
    def from_spec(spec: FactorSpec, *, trial_id: int, verdict: str, stage: str,
                  **kwargs: Any) -> "TrialRecord":
        """Build a record from a validated spec, filling the structural fields."""
        return TrialRecord(
            trial_id=trial_id,
            spec_hash=spec.spec_hash,
            verdict=verdict,
            stage=stage,
            theme=spec.theme,
            template=spec.template,
            hypothesis=spec.hypothesis,
            spec=spec.to_dict(),
            parent_spec_hash=spec.parent_spec_hash,
            n_grid_cells=spec.n_grid_cells,
            complexity=spec.complexity,
            **kwargs,
        )


# ======================================================================
# IO
# ======================================================================


def _path(path: Path | None = None) -> Path:
    return Path(path) if path is not None else config.TRIALS_PATH


def append_trial(record: TrialRecord, path: Path | None = None) -> TrialRecord:
    """Append one row. Creates the journal directory on first write."""
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(record.to_json() + "\n")
    logger.info(
        "trial %s %s at %s (%s)", record.trial_id, record.verdict,
        record.stage, record.spec_hash[:10],
    )
    return record


def iter_trials(path: Path | None = None) -> Iterator[TrialRecord]:
    """Yield every record, skipping blank lines. Missing file yields nothing."""
    p = _path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield TrialRecord.from_json(line)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("skipping malformed %s line %d: %s", p.name, lineno, exc)


def read_trials(path: Path | None = None) -> list[TrialRecord]:
    return list(iter_trials(path))


def next_trial_id(path: Path | None = None) -> int:
    ids = [r.trial_id for r in iter_trials(path)]
    return max(ids) + 1 if ids else 1


# ======================================================================
# Queries — the agent's global view
# ======================================================================


def check_if_tried(spec_or_hash: FactorSpec | str,
                   path: Path | None = None) -> TrialRecord | None:
    """Structural dedup. Returns the FIRST record for this structure, or None.

    Pre-backtest filter (A5): this is cheap, the backtest is not.
    """
    h = spec_or_hash.spec_hash if isinstance(spec_or_hash, FactorSpec) else str(spec_or_hash)
    for record in iter_trials(path):
        if record.spec_hash == h:
            return record
    return None


def get_sota_set(theme: str | None = None,
                 path: Path | None = None) -> list[TrialRecord]:
    """Accepted factors, best fitness first.

    Per the patterns note §C, SOTA is maintained *per theme* as well as
    globally, so the agent can ask "what is the best thing I have in `rates`".
    """
    accepted = [r for r in iter_trials(path) if r.verdict == "ACCEPT"]
    if theme is not None:
        accepted = [r for r in accepted if r.theme == theme]
    return sorted(
        accepted,
        key=lambda r: (r.metrics.get("fitness") is None, -(r.metrics.get("fitness") or 0.0)),
    )


def recent_diagnoses(n: int = 5, theme: str | None = None,
                     path: Path | None = None) -> list[TrialRecord]:
    """The last `n` records that carry a diagnosis, most recent first.

    Diagnoses are what gets fed forward into the next prompt — the log is for
    humans, the diagnosis is for the agent (patterns note Part 2, detail 3).
    """
    records = [r for r in iter_trials(path) if r.diagnosis]
    if theme is not None:
        records = [r for r in records if r.theme == theme]
    return list(reversed(records))[:n]


def failures_by_reason(theme: str | None = None,
                       path: Path | None = None) -> dict[str, int]:
    """Rejection-reason histogram — the evolution report's headline chart."""
    counts: dict[str, int] = {}
    for record in iter_trials(path):
        if record.verdict in ("ACCEPT", "REFINE"):
            continue
        if theme is not None and record.theme != theme:
            continue
        key = record.reject_reason or record.verdict
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def tried_hashes(path: Path | None = None) -> set[str]:
    return {r.spec_hash for r in iter_trials(path)}


def summary(path: Path | None = None) -> dict[str, Any]:
    """N_trials and the verdict breakdown. A9: never report Sharpe without this."""
    records = read_trials(path)
    verdicts: dict[str, int] = {}
    stages: dict[str, int] = {}
    for r in records:
        verdicts[r.verdict] = verdicts.get(r.verdict, 0) + 1
        stages[r.stage] = stages.get(r.stage, 0) + 1
    return {
        "n_trials": len(records),
        "n_unique_structures": len({r.spec_hash for r in records}),
        "verdicts": verdicts,
        "stages": stages,
        "n_accepted": verdicts.get("ACCEPT", 0),
    }


def reset(path: Path | None = None) -> None:
    """Delete the journal. Test helper only — never called by the loop."""
    p = _path(path)
    if p.exists():
        p.unlink()


__all__ = [
    "VERDICTS",
    "STAGES",
    "TrialRecord",
    "append_trial",
    "iter_trials",
    "read_trials",
    "next_trial_id",
    "check_if_tried",
    "get_sota_set",
    "recent_diagnoses",
    "failures_by_reason",
    "tried_hashes",
    "summary",
    "reset",
]
