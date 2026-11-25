"""The three tools the agent may call, and their dispatch. No LLM here (C1).

plan.md Phase 4 item 1 names them:

    check_if_tried(spec)   -> has this structure been seen, and what happened
    evaluate_spec(spec)    -> the full pipeline: dedup, complexity, align,
                              grid scan, gates, verdict
    write_journal(...)     -> the agent's own note on a trial

**Why `legs` is an array here and a dict everywhere else.** A `FactorSpec`
stores legs keyed by role (`direction`, `intensity`, ...), which is the right
shape for code. DeepSeek strict mode requires `additionalProperties: false`
and every property listed in `required`, so an object whose keys depend on the
chosen template cannot be expressed. The schema therefore asks for a list of
`{role, op, inputs, op_windows}` and `spec_from_payload` folds it back into the
dict. Validation is unchanged — `harness.spec.validate` still owns the rules.

Strict mode also rejects `minItems` / `maxItems` / `minLength` / `maxLength`,
so "a leg takes one or two inputs" cannot be a schema constraint. That rule
lives in `spec.validate` and comes back to the agent as a feedback message.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from harness import config
from harness.spec import (
    INDICATORS,
    OPERATORS,
    PRE_TRANSFORMS,
    TEMPLATES,
    FactorSpec,
    SpecError,
    SpecInput,
    SpecLeg,
)

logger = logging.getLogger(__name__)

THEME_NAMES = sorted({ind.theme for ind in INDICATORS.values()})
INDICATOR_CODES = sorted(INDICATORS)
OPERATOR_NAMES = sorted(OPERATORS)
PRE_NAMES = sorted(PRE_TRANSFORMS)
TEMPLATE_NAMES = sorted(TEMPLATES)
ROLE_NAMES = sorted({role for roles in TEMPLATES.values() for role in roles})


# ======================================================================
# Payload <-> FactorSpec
# ======================================================================


def spec_from_payload(payload: Mapping[str, Any]) -> FactorSpec:
    """Turn the agent's JSON into a validated FactorSpec.

    Raises `SpecError` with a message written to be handed straight back to the
    model — the loop does exactly that, which is the whole inner feedback path.
    """
    legs_in = payload.get("legs")
    if not isinstance(legs_in, list) or not legs_in:
        raise SpecError("`legs` must be a non-empty list of leg objects")

    legs: dict[str, SpecLeg] = {}
    for item in legs_in:
        if not isinstance(item, Mapping):
            raise SpecError(f"each leg must be an object, got {type(item).__name__}")
        role = item.get("role", "")
        if role in legs:
            raise SpecError(f'duplicate leg role "{role}"')
        inputs = []
        for raw in item.get("inputs") or []:
            inputs.append(SpecInput(
                indicator=raw.get("indicator", ""),
                pre=tuple(raw.get("pre") or ()),
                pre_windows=tuple(raw.get("pre_windows") or ()),
            ))
        legs[role] = SpecLeg(
            op=item.get("op", ""),
            inputs=tuple(inputs),
            op_windows=tuple(item.get("op_windows") or ()),
        )

    parent = payload.get("parent_spec_hash") or None
    return FactorSpec(
        theme=payload.get("theme", ""),
        template=payload.get("template", ""),
        legs=legs,
        hypothesis=payload.get("hypothesis", ""),
        parent_spec_hash=parent,
        mutation_note=payload.get("mutation_note", "") or "",
    )


def payload_from_spec(spec: FactorSpec) -> dict:
    """Inverse of `spec_from_payload` — used to echo a spec back to the agent."""
    return {
        "theme": spec.theme,
        "template": spec.template,
        "legs": [
            {
                "role": role,
                "op": leg.op,
                "inputs": [
                    {"indicator": i.indicator, "pre": list(i.pre),
                     "pre_windows": list(i.pre_windows)}
                    for i in leg.inputs
                ],
                "op_windows": list(leg.op_windows),
            }
            for role, leg in spec.legs.items()
        ],
        "hypothesis": spec.hypothesis,
        "parent_spec_hash": spec.parent_spec_hash or "",
        "mutation_note": spec.mutation_note,
    }


# ======================================================================
# JSON Schema
# ======================================================================


def _spec_schema() -> dict:
    """The FactorSpec schema, in the subset DeepSeek strict mode accepts."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["theme", "template", "legs", "hypothesis",
                     "parent_spec_hash", "mutation_note"],
        "properties": {
            "theme": {
                "type": "string", "enum": THEME_NAMES,
                "description": "Which of the five economic channels this factor belongs to.",
            },
            "template": {
                "type": "string", "enum": TEMPLATE_NAMES,
                "description": (
                    "Composition shape. The roles you must supply in `legs` are "
                    "fixed by this choice; see the system prompt's template table."
                ),
            },
            "legs": {
                "type": "array",
                "description": "One object per role required by the template.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "op", "inputs", "op_windows"],
                    "properties": {
                        "role": {"type": "string", "enum": ROLE_NAMES},
                        "op": {"type": "string", "enum": OPERATOR_NAMES},
                        "inputs": {
                            "type": "array",
                            "description": (
                                "One entry per operator input. Most operators take "
                                "one; corr_change, corr_slope and ratio_over_logsize "
                                "take two."
                            ),
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["indicator", "pre", "pre_windows"],
                                "properties": {
                                    "indicator": {"type": "string", "enum": INDICATOR_CODES},
                                    "pre": {
                                        "type": "array",
                                        "description": (
                                            "Pre-transforms applied in order, at most "
                                            f"{config.MAX_PRE_TRANSFORMS_PER_INPUT}. "
                                            "Use [] for none."
                                        ),
                                        "items": {"type": "string", "enum": PRE_NAMES},
                                    },
                                    "pre_windows": {
                                        "type": "array",
                                        "description": (
                                            "Window roles for pre-transforms that need "
                                            "one (only `rank` does). Use [] otherwise."
                                        ),
                                        "items": {"type": "string",
                                                  "enum": list(config.ROLE_NAMES)},
                                    },
                                },
                            },
                        },
                        "op_windows": {
                            "type": "array",
                            "description": (
                                "Window ROLE names, never numbers. Two slots bound to "
                                "the same role move together; different roles are "
                                "scanned independently over {5, 20, 60}."
                            ),
                            "items": {"type": "string", "enum": list(config.ROLE_NAMES)},
                        },
                    },
                },
            },
            "hypothesis": {
                "type": "string",
                "description": (
                    "One or two sentences of economics: why this structure should "
                    "predict gold returns. It is scored for consistency with the "
                    "structure before any backtest runs, so do not write a story "
                    "that the legs do not implement."
                ),
            },
            "parent_spec_hash": {
                "type": "string",
                "description": "The spec you mutated, or \"\" if this is a fresh idea.",
            },
            "mutation_note": {
                "type": "string",
                "description": "What you changed relative to the parent, or \"\".",
            },
        },
    }


def tool_schemas(strict: bool = True) -> list[dict]:
    """The three function definitions, ready for the `tools` request field."""
    spec_param = _spec_schema()

    defs = [
        {
            "name": "check_if_tried",
            "description": (
                "Cheap structural lookup. Returns whether this exact structure has "
                "been evaluated before and, if so, what the verdict and diagnosis "
                "were. Call it before evaluate_spec when you are unsure — it costs "
                "nothing and a duplicate wastes a trial."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["spec"],
                "properties": {"spec": spec_param},
            },
        },
        {
            "name": "evaluate_spec",
            "description": (
                "Run the full deterministic pipeline on a structure: schema "
                "validation, structural dedup, complexity check, hypothesis "
                "alignment, window-grid backtest, signal dedup against the SOTA "
                "set, and the six gates. Returns a verdict (ACCEPT / REFINE / "
                "REJECT) with metrics and a diagnosis telling you what to change. "
                "This is the only call that costs a trial."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["spec"],
                "properties": {"spec": spec_param},
            },
        },
        {
            "name": "write_journal",
            "description": (
                "Record a short note of your own reasoning against a trial you "
                "have already evaluated — what you now believe about this corner "
                "of the search space. Your future self reads these."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["spec_hash", "note"],
                "properties": {
                    "spec_hash": {"type": "string",
                                  "description": "The spec_hash returned by evaluate_spec."},
                    "note": {"type": "string",
                             "description": "One or two sentences. Be specific."},
                },
            },
        },
    ]

    out = []
    for d in defs:
        fn: dict[str, Any] = {
            "name": d["name"],
            "description": d["description"],
            "parameters": d["parameters"],
        }
        if strict:
            # DeepSeek: "all `function` need to set the `strict` property to true"
            fn["strict"] = True
        out.append({"type": "function", "function": fn})
    return out


TOOL_NAMES = ("check_if_tried", "evaluate_spec", "write_journal")


# ======================================================================
# Dispatch
# ======================================================================


class ToolDispatcher:
    """Routes a tool call to a handler. The handlers live in `loop.py`.

    Keeping dispatch separate from the loop means `test_phase4.py` can drive
    the tools with no model and no network.
    """

    def __init__(self, handlers: Mapping[str, Callable[..., dict]]):
        unknown = set(handlers) - set(TOOL_NAMES)
        if unknown:
            raise ValueError(f"handler for unknown tool(s): {sorted(unknown)}")
        missing = set(TOOL_NAMES) - set(handlers)
        if missing:
            raise ValueError(f"no handler for tool(s): {sorted(missing)}")
        self.handlers = dict(handlers)

    def dispatch(self, name: str, arguments: Mapping[str, Any]) -> dict:
        """Call a handler. Never raises — errors come back as a tool result.

        A tool that throws would end the run; a tool that returns
        `{"error": ...}` lets the agent read the message and try again, which
        is the inner feedback loop plan.md asks for.
        """
        handler = self.handlers.get(name)
        if handler is None:
            return {"error": f'unknown tool "{name}". Available: {list(TOOL_NAMES)}'}
        try:
            return handler(**arguments)
        except SpecError as exc:
            return {"error": f"spec rejected: {exc}", "retryable": True}
        except TypeError as exc:
            return {"error": f"wrong arguments for {name}: {exc}", "retryable": True}
        except Exception as exc:                            # noqa: BLE001
            logger.exception("tool %s raised", name)
            return {"error": f"{type(exc).__name__}: {exc}", "retryable": False}


__all__ = [
    "INDICATOR_CODES",
    "OPERATOR_NAMES",
    "PRE_NAMES",
    "ROLE_NAMES",
    "TEMPLATE_NAMES",
    "THEME_NAMES",
    "TOOL_NAMES",
    "ToolDispatcher",
    "payload_from_spec",
    "spec_from_payload",
    "tool_schemas",
]
