"""① Specification Unit — FactorSpec schema, whitelists, canonical hash.

Schema version 4. ``plan.md`` §3 sketches a v3 spec with exactly two legs
(``direction`` / ``intensity``) and eight operators. Decomposing the 15 L0
factors in ``final_src_github/src/factors/build_factors.py`` showed v3 can
express **1 of the 15** (per-factor table in ``docs/pool_inventory.md``). v4 is
the smallest schema that covers all 15 while keeping every anti-pattern intact:

* A3 — the agent still emits JSON, never Python.
* A4 — no window *value* appears in a spec. A leg names window *roles*: which
  operator slots share one window and which move independently. Role names are
  structure; ``validator.py`` assigns values from ``config.WINDOW_VALUES``.
* Complexity control (AlphaAgent §1.5) survives as hard caps on pre-transforms
  per input, on distinct window roles, and on total grid cells.

A spec is a **structure**. Its hash covers ``template``, ``legs`` and the role
binding only — not ``theme``, ``hypothesis``, ``parent_spec_hash`` or
``mutation_note``. Two submissions describing the same computation collide even
if the agent retells the economics differently, which is what structural dedup
is for.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from harness import config

SPEC_VERSION = 4


class SpecError(ValueError):
    """Raised with a plain-language reason the agent can act on."""


# ======================================================================
# Themes
# ======================================================================

THEMES: tuple[str, ...] = ("dollar", "goldprice", "etf", "rates", "risk")


# ======================================================================
# Indicator whitelist
# ======================================================================
# Theme assignment follows final_src_github/docs/data_dictionary.md.
# `available_through` is from the staleness scan of data/aligned run 2026-11-21:
# the last date each series actually *changed*, ignoring the forward-fill
# padding the alignment script writes out to 2026-12-31.


@dataclass(frozen=True)
class Indicator:
    code: str
    label: str
    theme: str
    native_freq: str
    available_through: str


INDICATORS: dict[str, Indicator] = {
    ind.code: ind
    for ind in (
        Indicator("PMF0001", "10Y US Treasury nominal yield", "dollar", "daily", "2025-11-28"),
        Indicator("PMF0012", "US Dollar Index (DXY)", "dollar", "daily", "2025-11-28"),
        Indicator("PMF0011", "Au9995 Shanghai physical gold close", "goldprice", "daily", "2025-11-28"),
        Indicator("PMF0004", "COMEX gold futures close", "goldprice", "daily", "2025-11-28"),
        Indicator("PMF0020", "On/offshore gold futures basis", "goldprice", "daily", "2025-11-28"),
        Indicator("PMF0019", "Total gold ETF holdings", "etf", "daily", "2025-11-28"),
        Indicator("PMF0031", "SPDR gold monthly flow", "etf", "monthly", "2025-12-03"),
        Indicator("PMF0036", "SPDR gold monthly cash flow", "etf", "monthly", "2025-12-03"),
        Indicator("PMF0002", "10Y TIPS real yield", "rates", "daily", "2025-11-28"),
        Indicator("PMF0013", "US 10Y-2Y term spread", "rates", "daily", "2025-11-28"),
        Indicator("PMF0042", "US core CPI YoY, SA", "rates", "monthly", "2025-10-09"),
        Indicator("PMF0007", "Geopolitical Risk (GPR) index", "risk", "daily", "2026-02-13"),
        Indicator("PMF0018", "Gold 1M ATM implied volatility", "risk", "monthly", "2025-11-28"),
    )
}


# ======================================================================
# Pre-transform whitelist
# ======================================================================
# Applied to one input series, in order, before the operator sees it.
# `n_windows` is how many window roles the pre-transform consumes.

@dataclass(frozen=True)
class PreTransform:
    name: str
    n_windows: int
    expansion: str


PRE_TRANSFORMS: dict[str, PreTransform] = {
    p.name: p
    for p in (
        PreTransform("diff", 0, "x - ts_delay(x,1)"),
        PreTransform("pct_ret", 0, "x / ts_delay(x,1) - 1"),
        PreTransform("lag1", 0, "ts_delay(x,1)"),
        PreTransform("abs", 0, "abs(x)"),
        PreTransform("log_abs", 0, "abs(log(abs(x)))"),
        PreTransform("rank", 1, "ts_rank(x,n)"),
    )
}


# ======================================================================
# Operator whitelist
# ======================================================================
# `kind` is the output type. Templates accept only certain kinds per role,
# which is what stops the agent from multiplying two sign series and calling it
# direction × intensity.
#
# `ordered_slots=True` means the operator is degenerate unless its two windows
# differ (ma_cross with a == b is identically zero), so the two slots must bind
# to different roles and the grid only keeps cells where the second is longer.


@dataclass(frozen=True)
class Operator:
    name: str
    arity: int                 # number of indicator inputs
    n_windows: int             # how many window roles it consumes
    kind: str                  # sign | intensity | persistence | position | signal | level | raw
    expansion: str
    ordered_slots: bool = False


OPERATORS: dict[str, Operator] = {
    op.name: op
    for op in (
        Operator("level", 1, 0, "level", "x"),
        Operator("ma_diff", 1, 2, "sign",
                 "sign(ts_mean(x,a) - ts_delay(ts_mean(x,a),b))"),
        Operator("ma_cross", 1, 2, "sign",
                 "sign(ts_mean(x,a) - ts_mean(x,b))", ordered_slots=True),
        Operator("ma_spread_abs", 1, 2, "intensity",
                 "abs(ts_mean(x,a) - ts_mean(x,b))", ordered_slots=True),
        Operator("abs_pct_change", 1, 1, "intensity",
                 "abs(x / ts_delay(x,n) - 1)"),
        Operator("abs_zscore", 1, 1, "intensity",
                 "abs((x - ts_mean(x,n)) / ts_std(x,n))"),
        Operator("linreg_slope", 1, 1, "raw",
                 "ts_linreg_slope(x,n)"),
        Operator("aroon_balance", 1, 1, "signal",
                 "ts_aroon_balance(x,n)"),
        Operator("path_efficiency", 1, 1, "persistence",
                 "abs(x - ts_delay(x,n)) / ts_sum(abs(x - ts_delay(x,1)),n)"),
        Operator("autocorr_abs", 1, 1, "persistence",
                 "abs(ts_corr_binary(x, ts_delay(x,1), n))"),
        Operator("boll_position", 1, 1, "position",
                 "(x - (ts_mean(x,n) - ts_std(x,n))) / (2*ts_std(x,n)) - 0.5"),
        Operator("corr_change", 2, 2, "raw",
                 "c = ts_corr_binary(x,y,n); c - ts_delay(c,m)"),
        Operator("corr_slope", 2, 2, "raw",
                 "ts_linreg_slope(ts_corr_binary(x,y,n), m)"),
        Operator("ratio_over_logsize", 2, 1, "intensity",
                 "ts_mean(x / abs(log(abs(y))), n)"),
    )
}


# ======================================================================
# Composition templates
# ======================================================================
# Role -> operator kinds accepted in that role. Mirrors the five apply_*
# templates in the read-only build_factors.py (C3).

TEMPLATES: dict[str, dict[str, frozenset[str]]] = {
    "identity": {
        "value": frozenset({"raw", "signal", "intensity"}),
    },
    "risk_adjusted": {
        "numerator": frozenset({"raw", "intensity"}),
        "risk": frozenset({"level"}),
    },
    "direction_x_intensity": {
        "direction": frozenset({"sign"}),
        "intensity": frozenset({"intensity"}),
    },
    "signal_x_persistence": {
        "signal": frozenset({"sign", "signal"}),
        "persistence": frozenset({"persistence"}),
    },
    "direction_x_position": {
        "direction": frozenset({"sign"}),
        "position": frozenset({"position"}),
    },
}


# ======================================================================
# Data structures
# ======================================================================


@dataclass(frozen=True)
class SpecInput:
    """One indicator feeding one leg, with its ordered pre-transforms.

    `pre_windows` names one window role per window-consuming pre-transform,
    in order. Omit it and every such slot binds to "w1".
    """

    indicator: str
    pre: tuple[str, ...] = ()
    pre_windows: tuple[str, ...] = ()

    def n_pre_windows(self) -> int:
        return sum(PRE_TRANSFORMS[p].n_windows for p in self.pre if p in PRE_TRANSFORMS)

    def resolved_pre_windows(self) -> tuple[str, ...]:
        if self.pre_windows:
            return tuple(self.pre_windows)
        return ("w1",) * self.n_pre_windows()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"indicator": self.indicator, "pre": list(self.pre)}
        if self.pre_windows:
            d["pre_windows"] = list(self.pre_windows)
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SpecInput":
        if not isinstance(d, Mapping):
            raise SpecError(
                'an input must be an object like {"indicator": "PMF0012", '
                f'"pre": ["diff"]}}, got {d!r}'
            )
        if "indicator" not in d:
            raise SpecError('an input is missing the required "indicator" field')
        return SpecInput(
            indicator=str(d["indicator"]),
            pre=tuple(d.get("pre") or ()),
            pre_windows=tuple(d.get("pre_windows") or ()),
        )


@dataclass(frozen=True)
class SpecLeg:
    """One operator applied to its inputs, filling one template role.

    `op_windows` names one window role per operator window slot, in order.
    Omit it and the default binding applies: "w1" for every slot, except
    ordered-slot operators which default to ("w1", "w2").
    """

    op: str
    inputs: tuple[SpecInput, ...]
    op_windows: tuple[str, ...] = ()

    def resolved_op_windows(self) -> tuple[str, ...]:
        if self.op_windows:
            return tuple(self.op_windows)
        operator = OPERATORS.get(self.op)
        if operator is None:
            return ()
        if operator.ordered_slots and operator.n_windows == 2:
            return ("w1", "w2")
        return ("w1",) * operator.n_windows

    def all_roles(self) -> tuple[str, ...]:
        roles: list[str] = []
        for inp in self.inputs:
            roles.extend(inp.resolved_pre_windows())
        roles.extend(self.resolved_op_windows())
        return tuple(roles)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op, "inputs": [i.to_dict() for i in self.inputs]}
        if self.op_windows:
            d["op_windows"] = list(self.op_windows)
        return d

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SpecLeg":
        if not isinstance(d, Mapping):
            raise SpecError(
                f'a leg must be an object like {{"op": "...", "inputs": [...]}}, got {d!r}'
            )
        if "op" not in d:
            raise SpecError('a leg is missing the required "op" field')
        raw_inputs = d.get("inputs")
        if raw_inputs is None:
            raise SpecError(f'leg "{d["op"]}" is missing the required "inputs" list')
        return SpecLeg(
            op=str(d["op"]),
            inputs=tuple(SpecInput.from_dict(i) for i in raw_inputs),
            op_windows=tuple(d.get("op_windows") or ()),
        )


@dataclass(frozen=True)
class FactorSpec:
    """A factor *structure*. Carries window roles but no window values (A4)."""

    theme: str
    template: str
    legs: dict[str, SpecLeg]
    hypothesis: str
    parent_spec_hash: str | None = None
    mutation_note: str = ""
    spec_version: int = SPEC_VERSION
    _hash: str = field(default="", init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate(self)
        object.__setattr__(self, "_hash", _compute_hash(self.template, self.legs))

    # -- serialisation ------------------------------------------------

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "FactorSpec":
        if not isinstance(d, Mapping):
            raise SpecError(f"a spec must be a JSON object, got {type(d).__name__}")
        for required in ("theme", "template", "legs", "hypothesis"):
            if required not in d:
                raise SpecError(
                    f'spec is missing the required field "{required}". '
                    "A spec needs theme, template, legs and hypothesis."
                )
        raw_legs = d["legs"]
        if not isinstance(raw_legs, Mapping):
            raise SpecError(
                '"legs" must be an object mapping role -> leg, e.g. '
                '{"direction": {...}, "intensity": {...}}'
            )
        return FactorSpec(
            theme=str(d["theme"]),
            template=str(d["template"]),
            legs={str(role): SpecLeg.from_dict(leg) for role, leg in raw_legs.items()},
            hypothesis=str(d["hypothesis"]),
            parent_spec_hash=d.get("parent_spec_hash"),
            mutation_note=str(d.get("mutation_note", "")),
        )

    @staticmethod
    def from_json(text: str) -> "FactorSpec":
        try:
            return FactorSpec.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise SpecError(f"spec is not valid JSON: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "theme": self.theme,
            "template": self.template,
            "legs": {role: leg.to_dict() for role, leg in sorted(self.legs.items())},
            "hypothesis": self.hypothesis,
            "parent_spec_hash": self.parent_spec_hash,
            "mutation_note": self.mutation_note,
        }

    # -- derived ------------------------------------------------------

    @property
    def spec_hash(self) -> str:
        return self._hash

    @property
    def indicators(self) -> tuple[str, ...]:
        """Every indicator the spec touches, in canonical order."""
        seen: list[str] = []
        for _role, leg in sorted(self.legs.items()):
            for inp in leg.inputs:
                if inp.indicator not in seen:
                    seen.append(inp.indicator)
        return tuple(seen)

    @property
    def window_roles(self) -> tuple[str, ...]:
        """Distinct window roles the validator must scan, in config order."""
        used: set[str] = set()
        for leg in self.legs.values():
            used.update(leg.all_roles())
        return tuple(r for r in config.ROLE_NAMES if r in used)

    @property
    def n_grid_cells(self) -> int:
        return len(window_grid(self))

    @property
    def complexity(self) -> int:
        """Parameter-count proxy: legs + window slots + pre-transforms."""
        n = len(self.legs)
        for leg in self.legs.values():
            n += len(leg.all_roles())
            for inp in leg.inputs:
                n += len(inp.pre)
        return n

    @property
    def structure_label(self) -> str:
        """Compact one-line rendering, for prompts, logs and the report."""
        parts = []
        for role, leg in sorted(self.legs.items()):
            args = ", ".join(
                ("+".join(inp.pre) + "(" + inp.indicator + ")") if inp.pre else inp.indicator
                for inp in leg.inputs
            )
            parts.append(f"{role}={leg.op}({args})")
        return f"{self.template}[" + " | ".join(parts) + "]"


# ======================================================================
# Window grid (A4 lives here, not in the spec)
# ======================================================================


def window_grid(spec: FactorSpec) -> list[dict[str, int]]:
    """Every valid role -> window assignment for this structure.

    Cells that make an ordered-slot operator degenerate (its second window not
    strictly longer than its first) are dropped, so the validator never wastes
    a backtest on an identically-zero series.
    """
    roles = spec.window_roles
    if not roles:
        return [{}]

    cells: list[dict[str, int]] = []
    for values in itertools.product(config.WINDOW_VALUES, repeat=len(roles)):
        assignment = dict(zip(roles, values))
        if _assignment_is_degenerate(spec, assignment):
            continue
        cells.append(assignment)
    return cells


def _assignment_is_degenerate(spec: FactorSpec, assignment: Mapping[str, int]) -> bool:
    for leg in spec.legs.values():
        operator = OPERATORS[leg.op]
        if not operator.ordered_slots:
            continue
        slots = leg.resolved_op_windows()
        if len(slots) == 2 and assignment[slots[0]] >= assignment[slots[1]]:
            return True
    return False


# ======================================================================
# Validation
# ======================================================================


def validate(spec: FactorSpec) -> None:
    """Raise SpecError with a reason the agent can act on, or return None."""
    if spec.theme not in THEMES:
        raise SpecError(f'unknown theme "{spec.theme}". Pick one of: {", ".join(THEMES)}.')

    if spec.template not in TEMPLATES:
        raise SpecError(
            f'unknown template "{spec.template}". Pick one of: '
            f'{", ".join(sorted(TEMPLATES))}.'
        )

    if not str(spec.hypothesis).strip():
        raise SpecError(
            "hypothesis is empty. Every spec must state the economic reason it should "
            "work BEFORE it is evaluated — the structure is later scored against it."
        )

    required_roles = TEMPLATES[spec.template]
    got, want = set(spec.legs), set(required_roles)
    if got != want:
        missing = ", ".join(sorted(want - got)) or "none"
        extra = ", ".join(sorted(got - want)) or "none"
        raise SpecError(
            f'template "{spec.template}" needs exactly these legs: '
            f'{", ".join(sorted(want))}. Missing: {missing}. Unexpected: {extra}.'
        )

    for role, leg in spec.legs.items():
        _validate_leg(spec.template, role, leg, required_roles[role])

    roles = spec.window_roles
    if len(roles) > config.MAX_WINDOW_ROLES:
        raise SpecError(
            f"this structure uses {len(roles)} independent window roles "
            f"({', '.join(roles)}), over the cap of {config.MAX_WINDOW_ROLES}. "
            "Tie some slots to the same role."
        )

    cells = len(window_grid(spec))
    if cells == 0:
        raise SpecError(
            "no valid window assignment exists for this structure — an ordered-slot "
            "operator (ma_cross / ma_spread_abs) has both windows bound to the same "
            "role, which makes it identically zero. Bind them to different roles."
        )
    if cells > config.MAX_GRID_CELLS:
        raise SpecError(
            f"this structure implies a {cells}-cell window grid, over the "
            f"{config.MAX_GRID_CELLS}-cell cap. Use fewer window-consuming operators."
        )


def _validate_leg(template: str, role: str, leg: SpecLeg,
                  allowed_kinds: frozenset[str]) -> None:
    if leg.op not in OPERATORS:
        raise SpecError(
            f'leg "{role}": unknown operator "{leg.op}". '
            f'Allowed operators: {", ".join(sorted(OPERATORS))}.'
        )
    operator = OPERATORS[leg.op]

    if operator.kind not in allowed_kinds:
        fits = sorted(name for name, o in OPERATORS.items() if o.kind in allowed_kinds)
        raise SpecError(
            f'leg "{role}" of template "{template}" needs an operator whose output is '
            f'{" or ".join(sorted(allowed_kinds))}, but "{leg.op}" produces '
            f'"{operator.kind}". Operators that fit here: {", ".join(fits)}.'
        )

    if len(leg.inputs) != operator.arity:
        raise SpecError(
            f'leg "{role}": operator "{leg.op}" takes {operator.arity} indicator '
            f"input(s), got {len(leg.inputs)}."
        )

    for position, inp in enumerate(leg.inputs):
        if inp.indicator not in INDICATORS:
            raise SpecError(
                f'leg "{role}" input {position}: unknown indicator "{inp.indicator}". '
                f'Allowed indicators: {", ".join(sorted(INDICATORS))}.'
            )
        if len(inp.pre) > config.MAX_PRE_TRANSFORMS_PER_INPUT:
            raise SpecError(
                f'leg "{role}" input {position}: {len(inp.pre)} pre-transforms, over the '
                f"cap of {config.MAX_PRE_TRANSFORMS_PER_INPUT} (complexity control)."
            )
        for pre in inp.pre:
            if pre not in PRE_TRANSFORMS:
                raise SpecError(
                    f'leg "{role}" input {position}: unknown pre-transform "{pre}". '
                    f'Allowed pre-transforms: {", ".join(sorted(PRE_TRANSFORMS))}.'
                )
        if inp.pre_windows and len(inp.pre_windows) != inp.n_pre_windows():
            raise SpecError(
                f'leg "{role}" input {position}: pre_windows has '
                f"{len(inp.pre_windows)} entries but the pre-transforms consume "
                f"{inp.n_pre_windows()} window(s)."
            )

    if leg.op_windows and len(leg.op_windows) != operator.n_windows:
        raise SpecError(
            f'leg "{role}": op_windows has {len(leg.op_windows)} entries but '
            f'"{leg.op}" consumes {operator.n_windows} window(s).'
        )

    for name in leg.all_roles():
        if name not in config.ROLE_NAMES:
            raise SpecError(
                f'leg "{role}": unknown window role "{name}". '
                f'Allowed roles: {", ".join(config.ROLE_NAMES)}.'
            )


# ======================================================================
# Canonical hash
# ======================================================================


def _canonical_structure(template: str, legs: Mapping[str, SpecLeg]) -> dict[str, Any]:
    """Structure-only, key-order-independent view used for hashing."""
    return {
        "spec_version": SPEC_VERSION,
        "template": template,
        "legs": {
            role: {
                "op": leg.op,
                # Input order is meaningful — ratio_over_logsize(x,y) is not
                # ratio_over_logsize(y,x) — so inputs are NOT sorted. Keys are.
                "inputs": [
                    {
                        "indicator": inp.indicator,
                        "pre": list(inp.pre),
                        "pre_windows": list(inp.resolved_pre_windows()),
                    }
                    for inp in leg.inputs
                ],
                "op_windows": list(leg.resolved_op_windows()),
            }
            for role, leg in sorted(legs.items())
        },
    }


def _compute_hash(template: str, legs: Mapping[str, SpecLeg]) -> str:
    payload = json.dumps(
        _canonical_structure(template, legs),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def spec_hash(spec: FactorSpec) -> str:
    return spec.spec_hash


__all__ = [
    "SPEC_VERSION",
    "SpecError",
    "THEMES",
    "INDICATORS",
    "PRE_TRANSFORMS",
    "OPERATORS",
    "TEMPLATES",
    "Indicator",
    "Operator",
    "PreTransform",
    "SpecInput",
    "SpecLeg",
    "FactorSpec",
    "window_grid",
    "validate",
    "spec_hash",
]
