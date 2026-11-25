"""② Synthesis Unit + the main research loop.

The loop is a sequential cycle, not a graph (A1), so it is a `for` statement.
One trial per iteration:

    global view -> agent proposes a spec -> pre-filters -> backtest -> gates
    -> diagnosis -> journal -> next iteration's global view

**Pre-filter order is load-bearing** (A5, plan.md Phase 4 item 3). The backtest
is the only expensive step, so everything that can reject a spec without one
runs first:

    1. schema validation   microseconds, and the error goes back to the agent
    2. structural dedup    one pass over trials.jsonl
    3. complexity          an integer comparison
    4. hypothesis align    one cheap LLM call
    5. grid scan + gates   ~6 seconds

**Resumability.** `next_trial_id()` reads the journal, so a run that dies at
trial 35 resumes at 35 — it does not restart. Over a 40-trial run the compute
is under a minute but the model calls are 25-60 minutes and real money, so
this is the single most valuable line in the file.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from harness import analyst, config, journal, prompts, tools
from harness.builder import build, load_indicators
from harness.llm import LLMClient, LLMError
from harness.spec import FactorSpec, SpecError
from harness.validator import ScanResult, load_returns, max_ic_vs_sota, scan

logger = logging.getLogger(__name__)


class LoopError(RuntimeError):
    pass


# ======================================================================
# Hypothesis alignment (plan.md Phase 4 item 4 — a first-class deliverable)
# ======================================================================


@dataclass(frozen=True)
class Alignment:
    score: int
    reason: str

    @property
    def passed(self) -> bool:
        return self.score >= config.ALIGNMENT_MIN_SCORE


def judge_alignment(spec: FactorSpec, client: LLMClient) -> Alignment:
    """Score hypothesis-vs-structure consistency with an independent call.

    This is the one thing in the project an LLM can do that random search
    cannot. A sweep can stumble onto a high Sharpe; it cannot say why, and a
    factor nobody can justify is a factor nobody can hold through a drawdown.
    Scoring the justification *before* the backtest is what stops the agent
    writing the story afterwards to fit the number.

    A failure to parse scores 3 ("unclear") rather than 0 — a flaky judge must
    not silently reject every spec.
    """
    reply = client.complete(
        [
            {"role": "system", "content": prompts.ALIGNMENT_SYSTEM},
            {"role": "user", "content": prompts.alignment_prompt(
                spec.structure_label, spec.theme, spec.hypothesis)},
        ],
        effort=config.LLM_JUDGE_EFFORT,
        label="align",
    )
    match = re.search(r"\{.*\}", reply.text, re.S)
    if not match:
        logger.warning("alignment judge returned no JSON: %r", reply.text[:200])
        return Alignment(3, "judge returned no parseable JSON; treated as neutral")
    try:
        data = json.loads(match.group(0))
        score = int(data.get("score", 3))
    except (json.JSONDecodeError, TypeError, ValueError):
        return Alignment(3, "judge JSON failed to parse; treated as neutral")
    return Alignment(max(1, min(5, score)), str(data.get("reason", "")))


# ======================================================================
# Trial outcome
# ======================================================================


@dataclass
class TrialOutcome:
    verdict: str
    stage: str
    spec: FactorSpec | None = None
    scan: ScanResult | None = None
    verdict_obj: analyst.Verdict | None = None
    alignment: Alignment | None = None
    max_ic: float = 0.0
    message: str = ""
    extra: dict = field(default_factory=dict)

    def to_tool_result(self) -> dict:
        """What the agent sees. Must be enough to act on, and nothing more."""
        out: dict[str, Any] = {"verdict": self.verdict, "stage": self.stage}
        if self.spec is not None:
            out["spec_hash"] = self.spec.spec_hash
            out["structure_label"] = self.spec.structure_label
        if self.alignment is not None:
            out["alignment"] = {"score": self.alignment.score,
                                "reason": self.alignment.reason}
        if self.scan is not None:
            out["metrics"] = {
                "fitness": round(self.scan.fitness, 4),
                "sharpe_median_train": round(self.scan.sharpe_median_train, 4),
                "sharpe_median_val": round(self.scan.sharpe_median_val, 4),
                "pass_rate": round(self.scan.pass_rate, 4),
                "n_cells": self.scan.n_cells,
                "ann_turnover_median": round(self.scan.ann_turnover_median, 4),
                "nan_ratio_median": round(self.scan.nan_ratio_median, 4),
                "max_ic_vs_sota": round(self.max_ic, 4),
                "sharpe_max_val_RECORD_ONLY": round(self.scan.sharpe_max_val, 4),
            }
        if self.verdict_obj is not None:
            out["failed_gates"] = list(self.verdict_obj.failed_gates)
            out["diagnosis"] = self.verdict_obj.diagnosis
            out["suggestions"] = list(self.verdict_obj.suggestions)
        if self.message:
            out["message"] = self.message
        out.update(self.extra)
        return out


# ======================================================================
# The engine behind the tools
# ======================================================================


class ResearchEngine:
    """Everything deterministic the agent can reach, plus the alignment judge.

    Holds the SOTA series cache so GATE_5 does not rebuild accepted factors on
    every trial.
    """

    def __init__(self, client: LLMClient, journal_path=None,
                 indicators=None, returns=None):
        self.client = client
        self.journal_path = journal_path
        self.inds = indicators if indicators is not None else load_indicators()
        self.returns = returns if returns is not None else load_returns()
        self._sota_series: dict[str, pd.Series] = {}
        self.notes: list[dict] = []

    # -- SOTA -------------------------------------------------------

    def sota_series(self) -> dict[str, pd.Series]:
        """Rebuild any accepted factor we have not cached yet.

        `record.spec` is the **journal** encoding, where `legs` is a dict keyed
        by role. `tools.spec_from_payload` parses the **tool-call** encoding,
        where `legs` is an array (strict mode forbids variable-key objects).
        The two differ only in that container type, so calling the wrong one
        raised `SpecError` rather than anything obvious — and the old
        `except: continue` turned that into an empty SOTA cache, a
        `max_ic_vs_sota` pinned at 0.0, and GATE_5 passing unconditionally for
        a whole 40-trial run (2026-11-24; the three accepted factors were
        re-checked afterwards at |IC| <= 0.28, so that run's verdicts stand).

        A parse failure here now raises. Skipping a SOTA record does not
        degrade gracefully: it silently disables a gate, and a crash is
        strictly cheaper than a run whose dedup never ran.
        """
        for record in journal.get_sota_set(path=self.journal_path):
            if record.spec_hash in self._sota_series:
                continue
            windows = record.metrics.get("representative_windows")
            try:
                spec = FactorSpec.from_dict(record.spec)
            except SpecError as exc:
                raise RuntimeError(
                    f"SOTA record {record.spec_hash[:10]} does not parse: {exc}. "
                    f"GATE_5 cannot be evaluated without it; refusing to "
                    f"continue with dedup silently disabled."
                ) from exc
            from harness.spec import window_grid
            cell = windows or window_grid(spec)[0]
            self._sota_series[record.spec_hash] = build(spec, cell, self.inds)
        return self._sota_series

    # -- tool: check_if_tried ---------------------------------------

    def check_if_tried(self, spec: Mapping[str, Any]) -> dict:
        parsed = tools.spec_from_payload(spec)
        hit = journal.check_if_tried(parsed, path=self.journal_path)
        if hit is None:
            return {"tried": False, "spec_hash": parsed.spec_hash,
                    "structure_label": parsed.structure_label}
        return {
            "tried": True, "spec_hash": parsed.spec_hash,
            "structure_label": parsed.structure_label,
            "previous_verdict": hit.verdict, "previous_trial_id": hit.trial_id,
            "diagnosis": hit.diagnosis,
        }

    # -- tool: write_journal ----------------------------------------

    def write_journal(self, spec_hash: str, note: str) -> dict:
        self.notes.append({"spec_hash": spec_hash, "note": note})
        return {"ok": True, "recorded": note[:80]}

    # -- tool: evaluate_spec ----------------------------------------

    def evaluate_spec(self, spec: Mapping[str, Any]) -> TrialOutcome:
        """The full pipeline, cheapest filter first (A5)."""
        # 1. schema — SpecError propagates to ToolDispatcher, which hands the
        #    message back to the agent verbatim. That IS the repair loop.
        parsed = tools.spec_from_payload(spec)

        # 2. structural dedup
        hit = journal.check_if_tried(parsed, path=self.journal_path)
        if hit is not None:
            return TrialOutcome(
                "DUPLICATE", "dedup_struct", spec=parsed,
                message=(f"This structure was already tried in round {hit.trial_id}, "
                         f"verdict {hit.verdict}. Diagnosis: {hit.diagnosis[:200]}"),
            )

        # 3. complexity
        if parsed.n_grid_cells > config.MAX_GRID_CELLS:
            return TrialOutcome(
                "INVALID", "complexity", spec=parsed,
                message=(f"The window grid has {parsed.n_grid_cells} cells, over the cap "
                         f"{config.MAX_GRID_CELLS}. Reduce the number of independent window roles."),
            )

        # 4. hypothesis alignment — one cheap model call, still before backtest
        align = judge_alignment(parsed, self.client)
        if not align.passed:
            return TrialOutcome(
                "MISALIGNED", "align", spec=parsed, alignment=align,
                message=(f"Hypothesis and structure are inconsistent ({align.score}/5): {align.reason} "
                         f"No backtest was run. Either change the structure to match the rationale, "
                         f"or state the economic mechanism this structure actually corresponds to."),
            )

        # 5. the expensive part
        result = scan(parsed, self.inds, self.returns)
        series = build(parsed, _first_cell(parsed), self.inds)
        max_ic = max_ic_vs_sota(series, self.sota_series())
        verdict = analyst.judge(result, max_ic_vs_sota=max_ic)

        return TrialOutcome(
            verdict.verdict, "gate", spec=parsed, scan=result,
            verdict_obj=verdict, alignment=align, max_ic=max_ic,
        )


def _first_cell(spec: FactorSpec) -> dict:
    from harness.spec import window_grid
    return window_grid(spec)[0]


# ======================================================================
# The loop
# ======================================================================


@dataclass
class RunSummary:
    started_at_trial: int
    finished_at_trial: int
    verdicts: dict[str, int]
    accepted: list[str]
    stopped_early: str = ""
    cost_usd: float = 0.0
    usage: dict = field(default_factory=dict)


def run(max_trials: int | None = None, *, client: LLMClient | None = None,
        journal_path=None, engine: ResearchEngine | None = None) -> RunSummary:
    """Run the research loop, resuming from wherever the journal left off."""
    max_trials = max_trials if max_trials is not None else config.MAX_TRIALS
    client = client or LLMClient()
    engine = engine or ResearchEngine(client, journal_path=journal_path)

    start = journal.next_trial_id(path=journal_path)
    if start > max_trials:
        logger.info("journal already holds %s trials; nothing to do", start - 1)
        return RunSummary(start, start - 1, journal.summary(journal_path)["verdicts"], [])

    system = prompts.system_prompt()
    schemas = tools.tool_schemas(strict=client.strict_tools)
    dispatcher = tools.ToolDispatcher({
        "check_if_tried": engine.check_if_tried,
        "write_journal": engine.write_journal,
        "evaluate_spec": lambda spec: engine.evaluate_spec(spec).to_tool_result(),
    })

    verdicts: dict[str, int] = {}
    accepted: list[str] = []
    consecutive_duplicates = 0
    stopped = ""
    trial_id = start - 1

    for trial_id in range(start, max_trials + 1):
        logger.info("─── trial %s/%s ───", trial_id, max_trials)
        context = prompts.round_prompt(
            trial_id,
            sota=journal.get_sota_set(path=journal_path),
            recent=journal.recent_diagnoses(4, path=journal_path),
            tried_count=len(journal.tried_hashes(path=journal_path)),
            failure_hist=journal.failures_by_reason(path=journal_path),
            budget_left=max_trials - trial_id + 1,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": context},
        ]

        outcome = _one_trial(messages, client, schemas, dispatcher, engine)

        record = _record_for(trial_id, outcome, engine)
        journal.append_trial(record, path=journal_path)
        verdicts[record.verdict] = verdicts.get(record.verdict, 0) + 1

        if outcome.verdict == "ACCEPT" and outcome.spec is not None:
            accepted.append(outcome.spec.spec_hash)
            engine._sota_series.pop(outcome.spec.spec_hash, None)
        if outcome.verdict == "REJECT" and outcome.verdict_obj and outcome.scan:
            analyst.record_failure(outcome.verdict_obj, outcome.scan)

        consecutive_duplicates = (
            consecutive_duplicates + 1 if outcome.verdict == "DUPLICATE" else 0
        )
        if consecutive_duplicates >= config.MAX_CONSECUTIVE_DUPLICATES:
            stopped = (
                f"{consecutive_duplicates} rounds in a row hit check_if_tried — the agent is spinning in place. "
                f"plan.md Phase 4 kill criterion: the global view is not reaching the prompt. "
                f"Fix round_prompt; do not paper over it by raising MAX_TRIALS."
            )
            logger.error(stopped)
            break

        logger.info("trial %s → %s  |  %s", trial_id, record.verdict, client.report())

    return RunSummary(
        started_at_trial=start, finished_at_trial=trial_id, verdicts=verdicts,
        accepted=accepted, stopped_early=stopped,
        cost_usd=client.cost_so_far(), usage=client.total.as_dict(),
    )


def _one_trial(messages: list[dict], client: LLMClient, schemas, dispatcher,
               engine: ResearchEngine) -> TrialOutcome:
    """Drive one trial's tool conversation until evaluate_spec returns."""
    last_outcome: TrialOutcome | None = None

    for step in range(config.MAX_TOOL_CALLS_PER_TRIAL):
        try:
            reply = client.complete(messages, schemas, label=f"synth.step{step}")
        except LLMError as exc:
            return TrialOutcome("ERROR", "validate", message=str(exc))

        if not reply.wants_tool:
            messages.append({"role": "assistant", "content": reply.text})
            messages.append({"role": "user", "content":
                             "Please submit a structure via the evaluate_spec tool; do not answer with text only."})
            continue

        messages.append({
            "role": "assistant", "content": reply.text or None,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"],
                              "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                for c in reply.tool_calls
            ],
        })

        for call in reply.tool_calls:
            if call["name"] == "evaluate_spec":
                try:
                    last_outcome = engine.evaluate_spec(call["arguments"].get("spec", {}))
                    result = last_outcome.to_tool_result()
                except SpecError as exc:
                    # The repair path: the agent sees exactly what it got wrong.
                    result = {"error": f"spec rejected: {exc}", "retryable": True}
                    last_outcome = None
            else:
                result = dispatcher.dispatch(call["name"], call["arguments"])

            messages.append({
                "role": "tool", "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

        if last_outcome is not None:
            return last_outcome

    return TrialOutcome(
        "ERROR", "validate",
        message=(f"No evaluable spec was produced within {config.MAX_TOOL_CALLS_PER_TRIAL} "
                 f"tool calls. The schema may have been rejected repeatedly."),
    )


def _record_for(trial_id: int, outcome: TrialOutcome,
                engine: ResearchEngine) -> journal.TrialRecord:
    """C7: one row per trial, whatever happened."""
    if outcome.spec is None:
        return journal.TrialRecord(
            trial_id=trial_id, spec_hash="", verdict=outcome.verdict,
            stage=outcome.stage, diagnosis=outcome.message,
            reject_reason=outcome.verdict.lower(),
        )

    fields: dict[str, Any] = {"source": "agent"}
    if outcome.scan is not None and outcome.verdict_obj is not None:
        fields.update(analyst.journal_fields(outcome.verdict_obj, outcome.scan))
        fields["metrics"]["max_ic_vs_sota"] = round(outcome.max_ic, 4)
        fields["metrics"]["representative_windows"] = _first_cell(outcome.spec)
    else:
        fields["diagnosis"] = outcome.message
        fields["reject_reason"] = outcome.verdict.lower()
    if outcome.alignment is not None:
        fields.setdefault("metrics", {})["alignment_score"] = outcome.alignment.score

    note = next((n["note"] for n in reversed(engine.notes)
                 if n["spec_hash"].startswith(outcome.spec.spec_hash[:10])), "")
    if note:
        fields["diagnosis"] = f"{fields.get('diagnosis', '')}\n[agent] {note}".strip()

    return journal.TrialRecord.from_spec(
        outcome.spec, trial_id=trial_id, verdict=outcome.verdict,
        stage=outcome.stage, **fields,
    )


__all__ = [
    "Alignment",
    "LoopError",
    "ResearchEngine",
    "RunSummary",
    "TrialOutcome",
    "judge_alignment",
    "run",
]
