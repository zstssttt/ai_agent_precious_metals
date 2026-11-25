#!/usr/bin/env python3
"""Run the agentic factor-research loop.

    python3 scripts/run_research.py --smoke          # 3 trials, cheap model
    python3 scripts/run_research.py --trials 40      # the real run
    python3 scripts/run_research.py --status         # what's in the journal
    python3 scripts/run_research.py --dry-run        # print the prompts, call nothing

The loop resumes: re-running after a crash picks up at the next trial id
rather than starting over. To start fresh, move `journal/trials.jsonl` aside.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import config, journal, prompts  # noqa: E402
from harness.llm import PROVIDERS, LLMClient, MissingAPIKey  # noqa: E402


def cmd_status() -> int:
    s = journal.summary()
    print(f"journal: {config.TRIALS_PATH}")
    if s["n_trials"] == 0:
        print("  empty — the next run starts at trial 1")
        return 0
    print(f"  trials            {s['n_trials']}")
    print(f"  unique structures {s['n_unique_structures']}")
    print(f"  verdicts          {s['verdicts']}")
    print(f"  stages            {s['stages']}")
    print(f"  next trial id     {journal.next_trial_id()}")
    sota = journal.get_sota_set()
    if sota:
        print(f"\n  SOTA ({len(sota)}):")
        for r in sota:
            print(f"    {r.spec_hash[:12]}  fitness={r.metrics.get('fitness')}  "
                  f"theme={r.theme}")
    hist = journal.failures_by_reason()
    if hist:
        print(f"\n  failures by reason: {hist}")
    return 0


def cmd_dry_run() -> int:
    sysp = prompts.system_prompt()
    print("=" * 70)
    print("SYSTEM PROMPT")
    print("=" * 70)
    print(sysp)
    print()
    print("=" * 70)
    print("ROUND PROMPT (trial 1)")
    print("=" * 70)
    print(prompts.round_prompt(
        1, sota=journal.get_sota_set(), recent=journal.recent_diagnoses(4),
        tried_count=len(journal.tried_hashes()),
        failure_hist=journal.failures_by_reason(),
        budget_left=config.MAX_TRIALS))
    print()
    cn = sum(0x4e00 <= ord(c) <= 0x9fff for c in sysp)   # count any CJK chars (token estimate)
    print(f"[system prompt ≈ {int(cn/1.5 + (len(sysp)-cn)/4)} tokens]")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=None,
                   help=f"budget ceiling, inclusive (default {config.MAX_TRIALS})")
    p.add_argument("--smoke", action="store_true",
                   help="3 trials on the cheap model — run this first")
    p.add_argument("--provider", default=None, choices=sorted(PROVIDERS))
    p.add_argument("--model", default=None)
    p.add_argument("--effort", default=None, choices=["low", "medium", "high"])
    p.add_argument("--status", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print the prompts and exit; no API key needed")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=config.LOG_FORMAT, datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    if args.status:
        return cmd_status()
    if args.dry_run:
        return cmd_dry_run()

    model = args.model
    trials = args.trials
    if args.smoke:
        trials = trials or (journal.next_trial_id() + 2)
        model = model or "deepseek-flash"
        print(f"smoke run: up to trial {trials} on {model}\n")

    try:
        client = LLMClient(provider=args.provider, model=model, effort=args.effort)
    except MissingAPIKey as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        print("See docs/setup_api.md for the full setup.", file=sys.stderr)
        return 2

    print(f"provider : {client.provider.name}")
    print(f"model    : {client.model}")
    print(f"endpoint : {client.base_url}")
    print(f"strict   : {client.strict_tools}")
    print(f"journal  : {config.TRIALS_PATH}")
    print(f"trace    : {config.TRACE_PATH}")
    print(f"resuming at trial {journal.next_trial_id()}\n")

    from harness.loop import run  # noqa: PLC0415 - imports pandas, keep --status fast

    summary = run(max_trials=trials, client=client)

    print("\n" + "=" * 70)
    print(f"trials {summary.started_at_trial}–{summary.finished_at_trial}")
    print(f"verdicts : {summary.verdicts}")
    print(f"accepted : {len(summary.accepted)}  {[h[:10] for h in summary.accepted]}")
    print(f"usage    : {summary.usage}")
    print(f"cost     : ${summary.cost_usd:.3f}")
    if summary.stopped_early:
        print(f"\n!! stopped early: {summary.stopped_early}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
