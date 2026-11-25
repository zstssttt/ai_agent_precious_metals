#!/usr/bin/env python3
"""Human-readable viewer for the two JSONL logs this harness writes.

    python3 scripts/view.py trials            # one card per trial
    python3 scripts/view.py trials --compact  # one-line-per-trial overview table
    python3 scripts/view.py trials --trial 7  # only round 7
    python3 scripts/view.py trace             # one block per LLM call
    python3 scripts/view.py trace --full      # no truncation, full prompt/reply
    python3 scripts/view.py trace --label synth  # only synthesis calls

Both files are JSONL (one JSON object per line), which is tiring to read raw.
This script is read-only and reformats them into cards. Paths default to
harness/config.py; override with --path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- color (auto-off when not a tty, forced off with --no-color) -------

class C:
    enabled = sys.stdout.isatty()

    @classmethod
    def wrap(cls, code: str, text: str) -> str:
        if not cls.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"


def dim(t): return C.wrap("2", t)
def bold(t): return C.wrap("1", t)
def red(t): return C.wrap("31", t)
def green(t): return C.wrap("32", t)
def yellow(t): return C.wrap("33", t)
def cyan(t): return C.wrap("36", t)


VERDICT_COLOR = {
    "ACCEPT": green, "REFINE": yellow, "REJECT": red,
    "DUPLICATE": dim, "INVALID": dim, "MISALIGNED": dim, "ERROR": red,
}

GATE_SHORT = {
    "GATE_1": "G1dir", "GATE_2": "G2decay", "GATE_3": "G3robust",
    "GATE_4": "G4turn", "GATE_5": "G5dedup", "GATE_6": "G6cover",
}


# ---- helpers -----------------------------------------------------------

def load(path: Path):
    if not path.exists():
        sys.exit(f"file not found: {path}")
    rows = []
    for i, line in enumerate(path.open(encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(dim(f"[line {i} failed to parse, skipped: {exc}]"), file=sys.stderr)
    return rows


def rule(label: str = "", width: int = 72) -> str:
    if not label:
        return dim("─" * width)
    left = f"── {label} "
    return dim(left + "─" * max(0, width - len(left)))


def truncate(text: str, limit: int, full: bool) -> str:
    text = text or ""
    if full or len(text) <= limit:
        return text
    return text[:limit] + dim(f" …(+{len(text) - limit} chars)")


def structure_str(spec: dict) -> str:
    """Render legs as a readable 'role=op(indicators)' string."""
    legs = spec.get("legs", {})
    parts = []
    for role, leg in legs.items():
        op = leg.get("op", "?")
        inds = []
        for inp in leg.get("inputs", []):
            code = inp.get("indicator", "?")
            pre = inp.get("pre", [])
            inds.append(f"{'→'.join(pre)}→{code}" if pre else code)
        parts.append(f"{role}={op}({', '.join(inds)})")
    return "  ×  ".join(parts)


# ---- trials.jsonl ------------------------------------------------------

def show_trials(rows, args):
    if args.trial is not None:
        rows = [r for r in rows if r.get("trial_id") == args.trial]
        if not rows:
            sys.exit(f"no trial {args.trial} in the journal")

    if args.compact:
        print(bold(f"{'#':>3}  {'verdict':<8} {'theme':<10} "
                   f"{'fitness':>8}  gates            parent"))
        for r in rows:
            v = r.get("verdict", "?")
            color = VERDICT_COLOR.get(v, str)
            gates = r.get("gates", {})
            marks = "".join(
                (green("✓") if gates.get(g, {}).get("passed") else red("✗"))
                for g in GATE_SHORT
            )
            fit = r.get("metrics", {}).get("fitness")
            fit_s = f"{fit:>8.3f}" if isinstance(fit, (int, float)) else f"{'—':>8}"
            parent = r.get("parent_spec_hash")
            parent_s = (parent[:8] if parent else dim("fresh"))
            print(f"{r.get('trial_id', '?'):>3}  {color(f'{v:<8}')} "
                  f"{r.get('theme', '?'):<10} {fit_s}  {marks}   {parent_s}")
        print()
        _trials_footer(rows)
        return

    for r in rows:
        _trial_card(r, args)
    _trials_footer(rows)


def _trial_card(r, args):
    v = r.get("verdict", "?")
    color = VERDICT_COLOR.get(v, str)
    tid = r.get("trial_id", "?")
    print()
    print(bold(f"━━━ Trial {tid}  ") + color(bold(v)) +
          bold(f"  ━━━ ") + dim(r.get("timestamp", "")))

    theme = r.get("theme", "?")
    template = r.get("template", "?")
    parent = r.get("parent_spec_hash")
    parent_s = (f"mutated from {parent[:10]}" if parent else "fresh structure")
    print(f"  theme {cyan(theme)}   template {cyan(template)}   {dim(parent_s)}")

    spec = r.get("spec", {})
    if spec.get("legs"):
        print(f"  structure {structure_str(spec)}")

    hyp = spec.get("hypothesis") or r.get("hypothesis")
    if hyp:
        print(dim("  hypothesis ") + truncate(hyp, 240, args.full))

    m = r.get("metrics", {})
    if m:
        pr = m.get("pass_rate")
        n = r.get("n_grid_cells")
        n_pass = int(round(pr * n)) if isinstance(pr, (int, float)) and n else "?"
        print(f"  metrics fitness {_num(m.get('fitness'))}"
              f"  |  train {_num(m.get('sharpe_median_train'))}"
              f" / val {_num(m.get('sharpe_median_val'))}"
              f"  |  pass_rate {_pct(pr)} ({n_pass}/{n})"
              f"  |  turnover {_num(m.get('ann_turnover_median'), 1)}")
        smax = m.get("sharpe_max_val")
        if isinstance(smax, (int, float)):
            print(dim(f"       sharpe_max {smax:.3f} (recorded only, not part of the verdict, C5)"))

    gates = r.get("gates", {})
    if gates:
        cells = []
        for g, short in GATE_SHORT.items():
            info = gates.get(g, {})
            ok = info.get("passed")
            mark = green("✓") if ok else red("✗")
            cells.append(f"{mark}{short}")
        print("  gates " + "  ".join(cells))

    diag = r.get("diagnosis")
    if diag:
        print(dim("  diagnosis ") + truncate(diag, 400, args.full))


def _trials_footer(rows):
    from collections import Counter
    verdicts = Counter(r.get("verdict", "?") for r in rows)
    print()
    print(rule())
    summary = "  ".join(
        VERDICT_COLOR.get(k, str)(f"{k} {v}") for k, v in verdicts.most_common()
    )
    print(f"{len(rows)} trials: {summary}")
    n_fresh = sum(1 for r in rows if not r.get("parent_spec_hash"))
    n_mut = len(rows) - n_fresh
    print(dim(f"of which {n_fresh} fresh, {n_mut} mutated from a prior round"
              f"{'  ← agent is not iterating on REFINE' if n_mut == 0 and len(rows) > 2 else ''}"))


def _num(x, digits=3):
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "—"


def _pct(x):
    return f"{x:.0%}" if isinstance(x, (int, float)) else "—"


# ---- llm_trace.jsonl ---------------------------------------------------

def show_trace(rows, args):
    if args.label:
        rows = [r for r in rows if args.label in (r.get("label") or "")]
        if not rows:
            sys.exit(f"no call in the trace whose label contains '{args.label}'")

    if args.finish:
        rows = [r for r in rows
                if r.get("response", {}).get("finish_reason") == args.finish]
        if not rows:
            print(green(f"no finish={args.finish} calls — clean."))
            return

    for r in rows:
        _trace_card(r, args)

    print()
    print(rule())
    total_r = sum(r.get("usage", {}).get("reasoning_tokens", 0) for r in rows)
    total_c = sum(r.get("usage", {}).get("completion_tokens", 0) for r in rows)
    n_len = sum(1 for r in rows
                if r.get("response", {}).get("finish_reason") == "length")
    print(f"{len(rows)} calls, "
          f"output {total_c:,} tok ({total_r:,} reasoning)")
    if n_len:
        print(red(f"⚠ {n_len} finish=length calls (truncated, likely wasted)"))
    else:
        print(green("✓ no finish=length truncation"))


def _trace_card(r, args):
    resp = r.get("response", {})
    fin = resp.get("finish_reason", "?")
    fin_s = red(fin) if fin == "length" else (
        cyan(fin) if fin == "tool_calls" else green(fin))
    u = r.get("usage", {})
    print()
    print(bold(f"[{r.get('ts', '')}] ") +
          bold(cyan(r.get("label", "?"))) +
          dim(f"  {r.get('model', '')}  attempt{r.get('attempt', '?')}"
              f"  {r.get('elapsed_s', '?')}s  ") +
          f"finish={fin_s}")
    print(dim(f"  tokens: {u.get('prompt_tokens', 0):,} in "
              f"({u.get('cached_tokens', 0):,} cached) / "
              f"{u.get('completion_tokens', 0):,} out "
              f"({u.get('reasoning_tokens', 0):,} reasoning)"))

    msgs = r.get("request", {}).get("messages", [])
    tools = r.get("request", {}).get("tools", [])
    print(rule(f"request  {len(msgs)} messages"
               f"{'  tools: ' + ','.join(t for t in tools if t) if tools else ''}"))
    for msg in msgs:
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, str):
            # the system prompt is long, so truncate it hard by default
            limit = 2000 if args.full else (150 if role == "system" else 400)
            print(f"  {bold('[' + role + ']')} {truncate(content, limit, args.full)}")
        else:
            print(f"  {bold('[' + role + ']')} {dim('(non-text content)')}")

    print(rule("response"))
    text = resp.get("text") or ""
    if text:
        print("  " + truncate(text, 2000 if args.full else 500, args.full))
    calls = resp.get("tool_calls", [])
    if calls:
        for c in calls:
            name = c.get("name", "?")
            arguments = c.get("arguments", {})
            print(f"  {yellow('→ calls ' + name)}")
            pretty = json.dumps(arguments, ensure_ascii=False,
                                indent=2 if args.full else None)
            print("    " + truncate(pretty, 600, args.full).replace("\n", "\n    "))
    if not text and not calls:
        print(dim("  (no text, no tool calls — most likely truncated)"))


# ---- main --------------------------------------------------------------

def main() -> int:
    from harness import config

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["trials", "trace"],
                   help="which file to view")
    p.add_argument("--path", type=Path, default=None,
                   help="override the default path")
    p.add_argument("--full", action="store_true",
                   help="no truncation, show full text")
    p.add_argument("--no-color", action="store_true",
                   help="disable color")
    p.add_argument("--trial", type=int, default=None,
                   help="(trials) show only one round")
    p.add_argument("--compact", action="store_true",
                   help="(trials) one-line-per-trial overview table")
    p.add_argument("--label", default=None,
                   help="(trace) only calls whose label contains this string, e.g. synth / align")
    p.add_argument("--finish", default=None,
                   help="(trace) only one finish_reason, e.g. length / stop / tool_calls")
    args = p.parse_args()

    if args.no_color:
        C.enabled = False

    if args.what == "trials":
        path = args.path or config.TRIALS_PATH
        show_trials(load(path), args)
    else:
        path = args.path or config.TRACE_PATH
        show_trace(load(path), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
