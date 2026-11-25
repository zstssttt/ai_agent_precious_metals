"""Prompt assembly. Text only — no model calls, no state.

Two prompts live here, and the split between them is the architecture, not a
formatting choice (plan.md §2, local view vs global view):

* `system_prompt()` is **frozen** for a whole run: the five themes, the
  whitelists, the rules file, the constraints. Because it never changes it
  sits at the front of every request and the provider's prompt cache serves
  it. Measured at ~2.6k tokens.
* `round_prompt()` is the **global view**: what has been tried, what is in the
  SOTA set, what the last diagnoses said. It changes every turn and therefore
  goes after the cached prefix.

The per-trial diagnosis the Analysis Unit writes is deliberately *not* merged
into the system prompt — `analyst.py` works from the local view of one
experiment, and keeping the two separate is what stops the agent over-fitting
to its most recent failure.
"""

from __future__ import annotations

from typing import Sequence

from harness import config
from harness.journal import TrialRecord
from harness.spec import INDICATORS, OPERATORS, PRE_TRANSFORMS, TEMPLATES

THEME_BLURBS = {
    "dollar": "Dollar & nominal-rate channel: gold is priced in USD; a stronger dollar / rising nominal rates usually weigh on the gold price.",
    "goldprice": "Price & spread channel: onshore-offshore spreads and the futures-spot basis reflect physical supply/demand and arbitrage pressure.",
    "etf": "Flow channel: ETF holdings and creation/redemption flows are a direct read on allocation money.",
    "rates": "Real-rate & inflation channel: gold pays no coupon, so the real rate is its opportunity cost; rising inflation expectations lift its store-of-value demand.",
    "risk": "Safe-haven channel: gold earns a haven premium when geopolitical risk or implied volatility rises.",
}


def _indicator_table() -> str:
    rows = []
    for code, ind in INDICATORS.items():
        flag = " ⚠monthly" if ind.native_freq == "monthly" else ""
        rows.append(f"  {code}  [{ind.theme:9}] {ind.label}{flag}")
    return "\n".join(rows)


def _operator_table() -> str:
    rows = []
    for name, op in sorted(OPERATORS.items()):
        slots = "1 input" if op.arity == 1 else f"{op.arity} inputs"
        wins = "no window" if op.n_windows == 0 else f"{op.n_windows} window role(s)"
        ordered = ", two slots must differ and the first must be shorter" if op.ordered_slots else ""
        rows.append(f"  {name:20} outputs {op.kind:12} {slots}, {wins}{ordered}")
    return "\n".join(rows)


def _template_table() -> str:
    rows = []
    for name, roles in TEMPLATES.items():
        parts = [f"{r}=<{'/'.join(sorted(k))}>" for r, k in roles.items()]
        rows.append(f"  {name:22} requires legs: {' , '.join(parts)}")
    return "\n".join(rows)


def _rules_text() -> str:
    path = config.RULES_DIR / "quant_hygiene.md"
    if not path.exists():
        return "(rules file missing)"
    return path.read_text(encoding="utf-8")


def system_prompt() -> str:
    """Frozen for the whole run. Keep it byte-stable or the cache never hits."""
    return f"""You are a researcher of gold-timing factors. Your job is to **propose factor structure hypotheses** and hand them to a deterministic engine for validation.

# What you do, and what you do not

You **only** produce a structured FactorSpec (via a tool call) and write one sentence of economic rationale for each structure.
You do **not** write code, do **not** run backtests, and do **not** pick parameters. Window parameters are not part of a spec — you only specify window **role names**
(w1/w2/w3), and the engine sweeps every combination over the {{5, 20, 60}} grid.

Within a spec, two slots bound to the same role move together; slots bound to different roles are scanned independently.
So the question "should I tune the window?" does not exist: it has already been fully swept. To change something, change the structure.

# The five themes (economic channels)

{chr(10).join(f"  {k:10} {v}" for k, v in THEME_BLURBS.items())}

# Indicator whitelist (only these 13 — do not invent other codes)

{_indicator_table()}

# Operator whitelist

{_operator_table()}

# Pre-transforms (at most {config.MAX_PRE_TRANSFORMS_PER_INPUT} per input, applied in order)

  {', '.join(sorted(PRE_TRANSFORMS))}
  Only `rank` needs a window role — put it in that input's pre_windows; leave the array empty otherwise.

# Templates (determine which legs you must provide, and each leg's operator output type)

{_template_table()}

  The angle brackets give the operator output type each role accepts. For example, the direction leg of
  direction_x_intensity only accepts operators that output sign (ma_diff / ma_cross).

# Quant-hygiene rules (must obey)

{_rules_text()}

# The six gates (how the engine adjudicates your spec)

  GATE_1  val-segment median Sharpe > {config.GATE_1_MIN_SHARPE_VAL}
  GATE_2  decay ratio val/train > {config.GATE_2_MIN_DECAY_RATIO} (a non-positive train fails outright)
  GATE_3  pass_rate > {config.GATE_3_MIN_PASS_RATE} (share of grid cells that clear the gates)
  GATE_4  median annualized turnover < {config.GATE_4_TURNOVER_CAP}
  GATE_5  max |IC| vs already-accepted factors < {config.GATE_5_MAX_IC_VS_SOTA}
  GATE_6  share of unusable values in the evaluation window < {config.GATE_6_MAX_NAN_RATIO}

  fitness = min(train median Sharpe, val median Sharpe), a conservative estimate.
  **The peak-cell Sharpe (sharpe_max) is recorded only and takes no part in any verdict** — do not judge a structure by it.
  What matters is pass_rate: a structure that holds in only 2 of 27 cells is a parameter accident, not an effect.

# Hard constraints

1. **You cannot see the test segment.** It is architecturally unreachable. Do not request it, infer it, or probe it indirectly with phrasing like "after 2024".
2. **The hypothesis must match the structure.** Each spec's hypothesis is scored independently (1-5),
   and a score below {config.ALIGNMENT_MIN_SCORE} is judged MISALIGNED with no backtest at all.
   Do not slap together a structure and then invent an economic story — that is exactly what this project guards against.
3. **Do not repeat a structure already tried.** When unsure, call check_if_tried first — it costs no budget.
4. **Call evaluate_spec once per round.** After you get the result you may call write_journal to record one sentence of your own judgement.

# How to use the diagnosis

The diagnosis returned by evaluate_spec names the failed gates and exactly what to change.
REFINE means the structure is worth mutating further — reuse parent_spec_hash and write clearly in mutation_note what you changed.
REJECT means change direction.
"""


def round_prompt(trial_id: int, *, sota: Sequence[TrialRecord],
                 recent: Sequence[TrialRecord],
                 tried_count: int, failure_hist: dict[str, int],
                 budget_left: int) -> str:
    """The global view, rebuilt every turn. Goes after the cached prefix."""
    lines = [f"# Round {trial_id} ({budget_left} rounds of budget left)", ""]

    if sota:
        lines.append(f"## Current SOTA set ({len(sota)} accepted factors)")
        for r in sota[:8]:
            fit = r.metrics.get("fitness")
            label = (r.spec.get("structure_label")
                     or f"{r.template}[{', '.join(sorted(r.spec.get('legs', {})))}]")
            lines.append(
                f"  {r.spec_hash[:10]}  theme={r.theme:9} fitness="
                f"{fit if fit is None else round(fit, 3)}  {label}"
            )
        lines.append("")
        lines.append("  A new structure must differ from these **in signal** (GATE_5), not just in wording.")
    else:
        lines.append("## Current SOTA set: empty. No factor has cleared the six gates yet.")
    lines.append("")

    lines.append(f"## {tried_count} structures tried so far")
    if failure_hist:
        top = ", ".join(f"{k}×{v}" for k, v in list(failure_hist.items())[:6])
        lines.append(f"  Failure-reason distribution: {top}")
    lines.append("")

    if recent:
        lines.append("## Recent rounds' diagnoses (this is what you should improve from)")
        for r in recent:
            lines.append(f"\n  [{r.verdict}] {r.spec_hash[:10]} theme={r.theme}")
            lines.append(f"  Hypothesis: {r.hypothesis[:120]}")
            lines.append(f"  Diagnosis: {r.diagnosis}")
        lines.append("")

    lines.append(
        "## This round's task\n"
        "Propose a new factor structure and call evaluate_spec. Think through the economic rationale first, "
        "then choose the structure — not the other way around. If the last round was REFINE, prefer mutating it "
        "(fill in parent_spec_hash); if several rounds in a row have failed on the same theme, switch themes."
    )
    return "\n".join(lines)


ALIGNMENT_SYSTEM = """You are a strict research reviewer. You do exactly one thing: judge whether a factor's **structure** truly matches its claimed **economic rationale**.

Score 1-5:
  5  the rationale precisely describes what the structure does; both the indicator choice and the operator choice are explained by it
  4  broadly matches, with one or two details the rationale does not cover
  3  right direction but vague; the rationale could be pasted onto a different structure
  2  the rationale clearly diverges from the structure (e.g. it talks about "rising volatility" but the structure has no volatility measure)
  1  the rationale is unrelated to the structure, or a generic after-the-fact story

Judge consistency only. Do **not** judge whether the factor will make money — that is the backtest's job.
Output a single JSON object: {"score": <integer 1-5>, "reason": "<one sentence, English>"}"""


def alignment_prompt(structure_label: str, theme: str, hypothesis: str) -> str:
    return (
        f"Theme: {theme}\n"
        f"Structure: {structure_label}\n"
        f"Claimed rationale: {hypothesis}\n\n"
        f"Is this rationale consistent with this structure?"
    )


__all__ = [
    "ALIGNMENT_SYSTEM",
    "THEME_BLURBS",
    "alignment_prompt",
    "round_prompt",
    "system_prompt",
]
