"""Phase 4 acceptance — Synthesis Unit + the loop.

plan.md asks for four things:
  1. a 3-trial run leaves 3 complete rows in trials.jsonl
  2. an illegal operator name comes back to the agent, which then corrects it
  3. round 2's prompt contains round 1's diagnosis (the feedback loop is closed)
  4. a hypothesis that contradicts its structure is stopped by the alignment gate

Nothing here touches the network. `FakeLLM` replays a script of tool calls, so
the tests exercise the real loop, the real dispatcher, the real engine and the
real gates — only the model is substituted. A test that needed an API key
would not run in CI and would cost money every time it did.
"""

from __future__ import annotations

import ast
import json

import pytest

from harness import config, journal, prompts, tools
from harness.llm import (
    PROVIDERS,
    UNSUPPORTED_SCHEMA_KEYS,
    Usage,
    retryable_errors,
)
from harness.l0_catalog import L0_SPECS
from harness.loop import ResearchEngine, judge_alignment, run
from harness.spec import FactorSpec, SpecError


# ======================================================================
# A model stand-in
# ======================================================================


class FakeLLM:
    """Replays scripted turns. Same surface as LLMClient, no network."""

    def __init__(self, script, align_score: int = 5):
        self.script = list(script)
        self.align_score = align_score
        self.calls: list[dict] = []
        self.strict_tools = True
        self.model = "fake"
        self.total = Usage()
        self.provider = PROVIDERS["deepseek"]

    def complete(self, messages, tools=None, *, effort=None, label=""):
        from harness.llm import LLMResponse

        self.calls.append({"label": label, "messages": [dict(m) for m in messages]})

        if label == "align":
            return LLMResponse(
                text=json.dumps({"score": self.align_score, "reason": "fake judge"}),
                tool_calls=(), finish_reason="stop", usage=Usage(calls=1),
            )

        if not self.script:
            raise AssertionError("FakeLLM ran out of script")
        turn = self.script.pop(0)
        if callable(turn):
            turn = turn(messages)
        return LLMResponse(
            text=turn.get("text", ""),
            tool_calls=tuple(turn.get("tool_calls", ())),
            finish_reason="tool_calls" if turn.get("tool_calls") else "stop",
            usage=Usage(prompt_tokens=100, completion_tokens=50, calls=1),
        )

    def cost_so_far(self) -> float:
        return 0.0

    def report(self) -> str:
        return "fake"

    # convenience for assertions
    def prompts_with(self, needle: str) -> list[dict]:
        return [c for c in self.calls
                if any(needle in str(m.get("content", "")) for m in c["messages"])]


def call(name: str, args: dict, cid: str = "c1") -> dict:
    return {"id": cid, "name": name, "arguments": args}


def payload(name: str = "pmf_051056", **over) -> dict:
    p = tools.payload_from_spec(L0_SPECS[name])
    p.update(over)
    return p


@pytest.fixture(scope="module")
def shared_data():
    from harness.builder import load_indicators
    from harness.validator import load_returns

    return load_indicators(), load_returns()


@pytest.fixture
def engine_factory(shared_data, tmp_path):
    inds, rets = shared_data

    def make(llm, path=None):
        return ResearchEngine(llm, journal_path=path or (tmp_path / "trials.jsonl"),
                              indicators=inds, returns=rets)

    return make


# ======================================================================
# 0b. Retry classification
# ======================================================================
#
# Pins the fix for the 2026-11-23 smoke failure: `httpx2` raised a local
# TypeError while decoding a 200, the old `except Exception` treated it as
# transport flake, and it was re-sent four times. The predicate is
# "would an identical resend plausibly succeed", not "is this serious".


@pytest.mark.parametrize("exc_name, should_retry", [
    # transient: server or network state, request itself was fine
    ("RateLimitError", True),            # 429
    ("InternalServerError", True),       # 5xx
    ("APIConnectionError", True),
    ("APITimeoutError", True),
    # deterministic: identical resend fails identically
    ("AuthenticationError", False),      # 401
    ("BadRequestError", False),          # 400
    ("PermissionDeniedError", False),    # 403
    ("NotFoundError", False),            # 404
    # 422 must reach loop.py as feedback, not be silently retried (C7)
    ("UnprocessableEntityError", False),
])
def test_retry_classification(exc_name, should_retry):
    openai = pytest.importorskip("openai")
    exc = getattr(openai, exc_name)
    assert issubclass(exc, retryable_errors()) is should_retry


def test_a_bug_in_our_own_process_is_never_retried():
    """The actual 2026-11-23 failure mode, as a regression test."""
    assert not issubclass(TypeError, retryable_errors())
    assert not issubclass(AttributeError, retryable_errors())
    assert not issubclass(KeyError, retryable_errors())


def test_the_4xx_parent_class_is_not_in_the_retry_tuple():
    """`APIStatusError` would isinstance-match every deterministic 4xx above."""
    openai = pytest.importorskip("openai")
    assert not issubclass(openai.AuthenticationError, retryable_errors())
    assert issubclass(openai.AuthenticationError, openai.APIStatusError), (
        "guard assumption changed: AuthenticationError no longer subclasses "
        "APIStatusError, so excluding the parent may no longer be sufficient"
    )


# ======================================================================
# 1. A short run leaves complete records
# ======================================================================


def test_three_trial_run_writes_three_complete_rows(engine_factory, tmp_path):
    path = tmp_path / "trials.jsonl"
    names = ["pmf_128453", "pmf_127111", "pmf_066409"]
    llm = FakeLLM([{"tool_calls": [call("evaluate_spec", {"spec": payload(n)})]}
                   for n in names])
    engine = engine_factory(llm, path)

    summary = run(max_trials=3, client=llm, journal_path=path, engine=engine)

    rows = journal.read_trials(path)
    assert len(rows) == 3
    assert [r.trial_id for r in rows] == [1, 2, 3]
    for r in rows:
        assert r.spec_hash and r.theme and r.template
        assert r.verdict in journal.VERDICTS
        assert r.stage in journal.STAGES
        assert r.diagnosis, f"trial {r.trial_id} has no diagnosis"
        assert r.metrics.get("fitness") is not None
        assert set(r.gates) == {"GATE_1", "GATE_2", "GATE_3",
                                "GATE_4", "GATE_5", "GATE_6"}
        assert r.timestamp
    assert summary.finished_at_trial == 3
    assert not summary.stopped_early


def test_run_resumes_instead_of_restarting(engine_factory, tmp_path):
    """A crash at trial 2 must not cost trial 1's model calls again."""
    path = tmp_path / "trials.jsonl"
    llm1 = FakeLLM([{"tool_calls": [call("evaluate_spec", {"spec": payload("pmf_128453")})]}])
    run(max_trials=1, client=llm1, journal_path=path, engine=engine_factory(llm1, path))
    assert len(journal.read_trials(path)) == 1

    llm2 = FakeLLM([{"tool_calls": [call("evaluate_spec", {"spec": payload("pmf_127111")})]}])
    summary = run(max_trials=2, client=llm2, journal_path=path,
                  engine=engine_factory(llm2, path))

    assert summary.started_at_trial == 2, "did not resume"
    rows = journal.read_trials(path)
    assert [r.trial_id for r in rows] == [1, 2]
    assert len(llm2.calls) < 6, "second run re-did the first trial"


def test_every_trial_is_journalled_even_when_it_fails(engine_factory, tmp_path):
    """C7: a spec that never reaches the backtest still gets a row."""
    path = tmp_path / "trials.jsonl"
    bad = payload()
    bad["legs"][0]["op"] = "no_such_operator"
    llm = FakeLLM([{"tool_calls": [call("evaluate_spec", {"spec": bad})]}] * 8)
    run(max_trials=1, client=llm, journal_path=path, engine=engine_factory(llm, path))

    rows = journal.read_trials(path)
    assert len(rows) == 1
    assert rows[0].verdict == "ERROR"
    assert rows[0].diagnosis


# ======================================================================
# 2. An illegal operator comes back to the agent
# ======================================================================


def test_illegal_operator_is_reported_back_and_can_be_corrected(
    engine_factory, tmp_path
):
    path = tmp_path / "trials.jsonl"
    bad = payload()
    bad["legs"][0]["op"] = "exponential_vibes"

    llm = FakeLLM([
        {"tool_calls": [call("evaluate_spec", {"spec": bad}, "bad")]},
        {"tool_calls": [call("evaluate_spec", {"spec": payload()}, "good")]},
    ])
    engine = engine_factory(llm, path)
    run(max_trials=1, client=llm, journal_path=path, engine=engine)

    # the error reached the model as a tool result, naming the offender
    tool_msgs = [m for c in llm.calls for m in c["messages"] if m.get("role") == "tool"]
    errors = [m["content"] for m in tool_msgs if "error" in m["content"]]
    assert errors, "the schema error never reached the model"
    assert "exponential_vibes" in errors[0]
    assert "ma_diff" in errors[0], "the error should list the legal operators"

    # and the corrected second attempt was evaluated
    rows = journal.read_trials(path)
    assert len(rows) == 1
    assert rows[0].verdict in ("ACCEPT", "REFINE", "REJECT")


def test_dispatcher_turns_exceptions_into_tool_results(engine_factory):
    """A throwing tool would end the run; a returned error lets it recover."""
    llm = FakeLLM([])
    engine = engine_factory(llm)
    d = tools.ToolDispatcher({
        "check_if_tried": engine.check_if_tried,
        "evaluate_spec": lambda spec: engine.evaluate_spec(spec).to_tool_result(),
        "write_journal": engine.write_journal,
    })
    out = d.dispatch("check_if_tried", {"spec": {"theme": "nope", "legs": []}})
    assert "error" in out and out["retryable"] is True

    out = d.dispatch("no_such_tool", {})
    assert "error" in out and "unknown tool" in out["error"]


def test_dispatcher_requires_exactly_the_three_tools(engine_factory):
    llm = FakeLLM([])
    engine = engine_factory(llm)
    with pytest.raises(ValueError, match="no handler"):
        tools.ToolDispatcher({"check_if_tried": engine.check_if_tried})


# ======================================================================
# 3. The feedback loop is closed
# ======================================================================


def test_round_two_prompt_contains_round_one_diagnosis(engine_factory, tmp_path):
    path = tmp_path / "trials.jsonl"
    llm = FakeLLM([
        {"tool_calls": [call("evaluate_spec", {"spec": payload("pmf_051056")})]},
        {"tool_calls": [call("evaluate_spec", {"spec": payload("pmf_104203")})]},
    ])
    run(max_trials=2, client=llm, journal_path=path, engine=engine_factory(llm, path))

    first = journal.read_trials(path)[0]
    assert first.diagnosis

    synth_calls = [c for c in llm.calls if c["label"].startswith("synth")]
    assert len(synth_calls) >= 2
    round_two = synth_calls[-1]["messages"][-1]["content"]
    assert first.diagnosis in round_two, (
        "round 2 did not see round 1's diagnosis — the feedback loop is open "
        "and the agent is searching blind"
    )
    assert first.spec_hash[:10] in round_two


def test_round_prompt_carries_the_sota_set(engine_factory, tmp_path):
    path = tmp_path / "trials.jsonl"
    llm = FakeLLM([
        {"tool_calls": [call("evaluate_spec", {"spec": payload("pmf_128453")})]},
        {"tool_calls": [call("evaluate_spec", {"spec": payload("pmf_127111")})]},
    ])
    run(max_trials=2, client=llm, journal_path=path, engine=engine_factory(llm, path))

    accepted = [r for r in journal.read_trials(path) if r.verdict == "ACCEPT"]
    if not accepted:
        pytest.skip("no ACCEPT in this pair; SOTA rendering covered by the unit test")
    last = [c for c in llm.calls if c["label"].startswith("synth")][-1]
    assert "SOTA" in last["messages"][-1]["content"]


def test_round_prompt_renders_without_history():
    text = prompts.round_prompt(1, sota=[], recent=[], tried_count=0,
                                failure_hist={}, budget_left=40)
    assert "Round 1" in text and "empty" in text


def test_system_prompt_is_frozen_within_a_run():
    """Byte-stable, or the provider cache never hits and every call pays full."""
    assert prompts.system_prompt() == prompts.system_prompt()


def test_system_prompt_carries_the_whitelists_and_the_rules():
    text = prompts.system_prompt()
    for code in tools.INDICATOR_CODES:
        assert code in text
    for op in tools.OPERATOR_NAMES:
        assert op in text
    assert "R1" in text and "R6" in text, "knowledge/rules did not make it in"
    assert "test" in text.lower()
    assert "sharpe_max" in text, "C5 must be stated to the agent"


def test_system_prompt_clears_the_cache_minimum():
    """DeepSeek and Opus cache prefixes of >=512 tokens; ours must exceed that."""
    text = prompts.system_prompt()
    cn = sum(0x4e00 <= ord(c) <= 0x9fff for c in text)   # count any CJK chars
    approx = int(cn / 1.5 + (len(text) - cn) / 4)
    assert approx > 512, f"system prompt is only ~{approx} tokens; too short to cache"


# ======================================================================
# 4. The alignment gate
# ======================================================================


def test_contradictory_hypothesis_is_rejected_before_the_backtest(
    engine_factory, tmp_path
):
    path = tmp_path / "trials.jsonl"
    nonsense = payload(hypothesis="This factor measures the duration risk of Japanese government bonds, unrelated to gold.")
    llm = FakeLLM([{"tool_calls": [call("evaluate_spec", {"spec": nonsense})]}],
                  align_score=1)
    run(max_trials=1, client=llm, journal_path=path, engine=engine_factory(llm, path))

    (row,) = journal.read_trials(path)
    assert row.verdict == "MISALIGNED"
    assert row.stage == "align", "alignment must run before the backtest"
    assert row.metrics.get("fitness") is None, "a misaligned spec must not be backtested"


def test_aligned_hypothesis_proceeds_to_the_backtest(engine_factory, tmp_path):
    path = tmp_path / "trials.jsonl"
    llm = FakeLLM([{"tool_calls": [call("evaluate_spec", {"spec": payload()})]}],
                  align_score=5)
    run(max_trials=1, client=llm, journal_path=path, engine=engine_factory(llm, path))
    (row,) = journal.read_trials(path)
    assert row.stage == "gate"
    assert row.metrics.get("alignment_score") == 5


@pytest.mark.parametrize("score,expected", [(1, False), (2, False), (3, True), (5, True)])
def test_alignment_threshold(score, expected, engine_factory):
    llm = FakeLLM([], align_score=score)
    result = judge_alignment(L0_SPECS["pmf_051056"], llm)
    assert result.passed is expected
    assert result.score == score


def test_unparseable_judge_output_is_neutral_not_fatal(engine_factory):
    """A flaky judge must not silently reject everything."""
    class Garbled(FakeLLM):
        def complete(self, messages, tools=None, *, effort=None, label=""):
            from harness.llm import LLMResponse
            return LLMResponse(text="I think it's fine, honestly.", tool_calls=(),
                               finish_reason="stop", usage=Usage(calls=1))

    result = judge_alignment(L0_SPECS["pmf_051056"], Garbled([]))
    assert result.score == 3 and result.passed


# ======================================================================
# Pre-filter ordering (A5)
# ======================================================================


def test_duplicate_is_caught_before_the_backtest(engine_factory, tmp_path):
    path = tmp_path / "trials.jsonl"
    llm = FakeLLM([
        {"tool_calls": [call("evaluate_spec", {"spec": payload()})]},
        {"tool_calls": [call("evaluate_spec", {"spec": payload()})]},   # same again
    ])
    engine = engine_factory(llm, path)
    run(max_trials=2, client=llm, journal_path=path, engine=engine)

    rows = journal.read_trials(path)
    assert rows[1].verdict == "DUPLICATE"
    assert rows[1].stage == "dedup_struct"
    assert rows[1].metrics.get("fitness") is None


def test_check_if_tried_is_free_and_informative(engine_factory, tmp_path):
    path = tmp_path / "trials.jsonl"
    llm = FakeLLM([{"tool_calls": [call("evaluate_spec", {"spec": payload()})]}])
    engine = engine_factory(llm, path)
    assert engine.check_if_tried(payload())["tried"] is False

    run(max_trials=1, client=llm, journal_path=path, engine=engine)
    again = engine.check_if_tried(payload())
    assert again["tried"] is True
    assert again["previous_trial_id"] == 1
    assert again["diagnosis"]


def test_stall_detector_stops_the_run(engine_factory, tmp_path):
    """plan.md kill criterion: repeated duplicates mean a broken prompt."""
    path = tmp_path / "trials.jsonl"
    n = config.MAX_CONSECUTIVE_DUPLICATES + 2
    llm = FakeLLM([{"tool_calls": [call("evaluate_spec", {"spec": payload()})]}] * n)
    summary = run(max_trials=n, client=llm, journal_path=path,
                  engine=engine_factory(llm, path))

    assert summary.stopped_early
    assert "spinning in place" in summary.stopped_early
    assert summary.finished_at_trial < n, "the run should have stopped before the budget"


# ======================================================================
# Schema compatibility with DeepSeek strict mode
# ======================================================================


def _walk(node, path="root"):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def test_every_object_is_closed_and_fully_required():
    """DeepSeek strict mode: all properties required, additionalProperties false."""
    for tool in tools.tool_schemas(strict=True):
        assert tool["function"]["strict"] is True
        for path, node in _walk(tool["function"]["parameters"]):
            if node.get("type") != "object":
                continue
            assert node.get("additionalProperties") is False, f"{path} is open"
            props = set(node.get("properties", {}))
            req = set(node.get("required", []))
            assert props == req, (
                f"{path}: strict mode needs every property in `required`; "
                f"missing {sorted(props - req)}"
            )


def test_schema_avoids_unsupported_keywords():
    for tool in tools.tool_schemas(strict=True):
        for path, node in _walk(tool["function"]["parameters"]):
            bad = set(node) & set(UNSUPPORTED_SCHEMA_KEYS)
            assert not bad, f"{path} uses {bad}, which strict mode rejects"


def test_strict_flag_is_omitted_when_disabled():
    for tool in tools.tool_schemas(strict=False):
        assert "strict" not in tool["function"]


def test_schema_enumerates_the_real_whitelists():
    spec_schema = None
    for tool in tools.tool_schemas():
        if tool["function"]["name"] == "evaluate_spec":
            spec_schema = tool["function"]["parameters"]["properties"]["spec"]
    assert spec_schema is not None
    leg = spec_schema["properties"]["legs"]["items"]["properties"]
    assert set(leg["op"]["enum"]) == set(tools.OPERATOR_NAMES)
    assert set(leg["inputs"]["items"]["properties"]["indicator"]["enum"]) == \
        set(tools.INDICATOR_CODES)
    assert set(spec_schema["properties"]["template"]["enum"]) == set(tools.TEMPLATE_NAMES)
    # A4: no window VALUE may appear anywhere in the schema
    blob = json.dumps(spec_schema, ensure_ascii=False)
    for role_enum in ("w1", "w2", "w3"):
        assert role_enum in blob
    assert '"enum": [5, 20, 60]' not in blob


@pytest.mark.parametrize("name", sorted(L0_SPECS))
def test_payload_round_trip_preserves_the_structure(name):
    original = L0_SPECS[name]
    restored = tools.spec_from_payload(tools.payload_from_spec(original))
    assert restored.spec_hash == original.spec_hash
    assert restored.structure_label == original.structure_label


def test_malformed_payload_raises_a_readable_spec_error():
    with pytest.raises(SpecError, match="non-empty list"):
        tools.spec_from_payload({"theme": "risk", "template": "identity", "legs": []})


# -- the two encodings must not be confused for each other --------------
#
# `legs` is an array in the tool-call encoding (strict mode forbids
# variable-key objects) and a dict in the journal encoding. Everything else
# is byte-identical, so passing one to the other's parser raises SpecError
# instead of anything that looks like a type error. On 2026-11-24 that cost a
# whole 40-trial run its GATE_5: `sota_series()` parsed journal rows with
# `spec_from_payload`, swallowed the SpecError, and left `max_ic_vs_sota`
# pinned at 0.0 for every trial.


@pytest.mark.parametrize("name", sorted(L0_SPECS))
def test_journal_encoding_round_trips_through_from_dict(name):
    """What `sota_series()` actually needs: journal row -> spec, hash intact."""
    original = L0_SPECS[name]
    restored = FactorSpec.from_dict(original.to_dict())
    assert restored.spec_hash == original.spec_hash
    assert restored.structure_label == original.structure_label


@pytest.mark.parametrize("name", sorted(L0_SPECS))
def test_the_two_encodings_are_not_interchangeable(name):
    """Pins the confusion itself, so a future refactor cannot re-merge them.

    If either direction ever stops raising, the two encodings have converged
    and the guard in `sota_series()` is no longer load-bearing — that is a
    deliberate change, and this test is where it should be noticed.
    """
    spec = L0_SPECS[name]
    with pytest.raises(SpecError):
        tools.spec_from_payload(spec.to_dict())           # dict legs -> array parser
    with pytest.raises(SpecError):
        FactorSpec.from_dict(tools.payload_from_spec(spec))   # array legs -> dict parser


def test_sota_rebuild_reads_the_journal_encoding(engine_factory, tmp_path):
    """End-to-end: an ACCEPTed row must come back as a usable series.

    The regression this pins is not "a warning was logged" but "GATE_5 had
    nothing to compare against".
    """
    path = tmp_path / "trials.jsonl"
    spec = L0_SPECS[sorted(L0_SPECS)[0]]
    journal.append_trial(
        journal.TrialRecord.from_spec(
            spec, trial_id=1, stage="gate", verdict="ACCEPT",
            metrics={"fitness": 0.3, "representative_windows": {"w1": 20, "w2": 20}},
            diagnosis="", gates=[],
        ),
        path=path,
    )
    engine = engine_factory(FakeLLM([]), path)

    series = engine.sota_series()

    assert spec.spec_hash in series, "accepted factor was not rebuilt"
    assert series[spec.spec_hash].notna().any(), "rebuilt series is all-NaN"


def test_an_unparseable_sota_record_raises_instead_of_disabling_the_gate(
        engine_factory, tmp_path):
    """A skipped SOTA record silently switches GATE_5 off — never degrade."""
    path = tmp_path / "trials.jsonl"
    spec = L0_SPECS[sorted(L0_SPECS)[0]]
    record = journal.TrialRecord.from_spec(
        spec, trial_id=1, stage="gate", verdict="ACCEPT",
        metrics={"fitness": 0.3}, diagnosis="", gates=[],
    )
    journal.append_trial(record, path=path)
    # Corrupt the *inner* spec, not the row: the row must still load as a
    # TrialRecord so execution actually reaches sota_series(). Corrupting the
    # row instead makes `read_trials` skip the line with only a warning, which
    # empties the SOTA set just as silently — see BACKLOG.
    raw = path.read_text(encoding="utf-8")
    assert spec.template in raw
    path.write_text(raw.replace(spec.template, "no_such_template"),
                    encoding="utf-8")
    engine = engine_factory(FakeLLM([]), path)

    with pytest.raises((RuntimeError, SpecError)):
        engine.sota_series()


# ======================================================================
# C1 — the deterministic half must stay model-free
# ======================================================================


@pytest.mark.parametrize("module", ["builder", "validator", "analyst", "spec",
                                    "journal", "tools", "config"])
def test_deterministic_modules_never_import_the_llm(module):
    """C1 as a parse, not a promise."""
    path = config.PROJECT_ROOT / "harness" / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = {"openai", "anthropic", "harness.llm", "httpx", "requests"}
    assert not (imported & banned), f"harness/{module}.py imports {imported & banned}"


def test_only_loop_and_llm_may_reach_a_model():
    """Whole-package sweep, so a new module cannot quietly add a client."""
    allowed = {"llm.py", "loop.py"}
    offenders = []
    for path in sorted((config.PROJECT_ROOT / "harness").glob("*.py")):
        if path.name in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            if any(m.split(".")[0] in ("openai", "anthropic") or m == "harness.llm"
                   for m in mods):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, f"model access outside {allowed}: {offenders}"


def test_agent_never_sees_a_test_window_metric(engine_factory, tmp_path):
    """C2 end to end: nothing in a tool result mentions the test split."""
    path = tmp_path / "trials.jsonl"
    llm = FakeLLM([{"tool_calls": [call("evaluate_spec", {"spec": payload()})]}])
    engine = engine_factory(llm, path)
    result = engine.evaluate_spec(payload()).to_tool_result()

    blob = json.dumps(result, ensure_ascii=False)
    assert "test" not in blob.lower()
    assert config.TEST_SPLIT[0] not in blob
    assert config.TEST_SPLIT[1] not in blob


def test_tool_result_marks_sharpe_max_as_record_only(engine_factory):
    """C5: the agent may see the peak, but the field name must say it decides nothing."""
    llm = FakeLLM([])
    engine = engine_factory(llm)
    result = engine.evaluate_spec(payload()).to_tool_result()
    assert "sharpe_max_val_RECORD_ONLY" in result["metrics"]
    assert "sharpe_max_val" not in result["metrics"]
