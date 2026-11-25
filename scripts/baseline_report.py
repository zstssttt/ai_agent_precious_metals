#!/usr/bin/env python3
"""Re-run the 15 hand-built L0 factors under the CURRENT rules (A7).

    python3 scripts/baseline_report.py                # table to stdout
    python3 scripts/baseline_report.py --json out.json

Why this script exists
----------------------
A7 forbids comparing the agent against the 0.939 in the upstream README: that
number comes from a full-sample direction fit with no train/val split. The only
admissible human baseline is the same 15 structures scored by the same engine,
under the same split, with the same gates and the same `decide()` rule.

"Same rule" is load-bearing and is why this cannot be a number copied from an
earlier run: `decide()` changed on 2026-11-24 (`pass_rate == 0` plus a
non-positive median in EITHER split now REJECTs, where it previously keyed off
`val` alone). Every verdict produced before that change is in a different
regime, agent and human alike.

GATE_5 and ordering
-------------------
GATE_5 compares a candidate against the factors already accepted, so it is
path-dependent: it needs an order. The agent's order is its trial sequence.
The human catalog has no intrinsic order, so this script reports both:

* `independent` — gates 1/2/3/4/6 only, every factor judged on its own. This is
  the ceiling: how many structures are individually sound.
* `sequential`  — catalog order, accumulating a SOTA set exactly as the loop
  does. This is the apples-to-apples number against the agent's ACCEPT count.

Reporting only `sequential` would understate the human pool (a near-duplicate
pair loses whichever member comes second); reporting only `independent` would
overstate it against an agent that did pay the dedup cost. Both, always.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import analyst, config, setup_logging  # noqa: E402
from harness.builder import build, load_indicators  # noqa: E402
from harness.l0_catalog import L0_SPECS, L0_SURVIVORS  # noqa: E402
from harness.spec import window_grid  # noqa: E402
from harness.validator import load_returns, max_ic_vs_sota, scan  # noqa: E402


def evaluate() -> list[dict]:
    inds = load_indicators()
    returns = load_returns()
    rows: list[dict] = []
    sota: dict[str, "object"] = {}

    for name, spec in L0_SPECS.items():
        res = scan(spec, inds, returns)

        # independent: no SOTA to compare against
        solo = analyst.judge(res, max_ic_vs_sota=0.0)

        # sequential: compare against everything accepted so far, the same way
        # loop.ResearchEngine does, at the same representative cell.
        fields = analyst.journal_fields(solo, res)
        cell = fields.get("representative_windows") or window_grid(spec)[0]
        series = build(spec, cell, inds)
        ic = max_ic_vs_sota(series, sota) if sota else 0.0
        seq = analyst.judge(res, max_ic_vs_sota=ic)
        if seq.verdict == "ACCEPT":
            sota[spec.spec_hash] = series

        rows.append({
            "factor": name,
            "published_survivor": name in L0_SURVIVORS,
            "theme": spec.theme,
            "template": spec.template,
            "n_cells": res.n_cells,
            "pass_rate": round(res.pass_rate, 4),
            "sharpe_median_train": round(res.sharpe_median_train, 4),
            "sharpe_median_val": round(res.sharpe_median_val, 4),
            "sharpe_max_val_RECORD_ONLY": round(res.sharpe_max_val, 4),
            "ann_turnover_median": round(res.ann_turnover_median, 4),
            "nan_ratio_median": round(res.nan_ratio_median, 6),
            "fitness": round(solo.fitness, 4),
            "max_ic_vs_sota": round(ic, 4),
            "verdict_independent": solo.verdict,
            "verdict_sequential": seq.verdict,
            "failed_independent": list(solo.failed_gates),
            "failed_sequential": list(seq.failed_gates),
            "representative_windows": cell,
            "spec_hash": spec.spec_hash[:10],
        })
    return rows


def print_table(rows: list[dict]) -> None:
    hdr = ("factor", "theme", "cells", "pass_rate", "train", "val",
           "fitness", "max_ic", "independent", "sequential")
    print(f"{'factor':12} {'theme':10} {'cells':>5} {'pass':>6} "
          f"{'train':>7} {'val':>7} {'fitness':>8} {'max_ic':>7}  "
          f"{'independent':11} {'sequential':11}")
    print("-" * 104)
    for r in rows:
        print(f"{r['factor']:12} {r['theme']:10} {r['n_cells']:5} "
              f"{r['pass_rate']:6.3f} {r['sharpe_median_train']:+7.3f} "
              f"{r['sharpe_median_val']:+7.3f} {r['fitness']:+8.3f} "
              f"{r['max_ic_vs_sota']:7.3f}  "
              f"{r['verdict_independent']:11} {r['verdict_sequential']:11}")

    def tally(key: str) -> dict:
        out: dict[str, int] = {}
        for r in rows:
            out[r[key]] = out.get(r[key], 0) + 1
        return out

    print()
    print(f"N_factors               : {len(rows)}   (N_trials = {len(rows)}, A9)")
    print(f"verdicts (independent)  : {tally('verdict_independent')}")
    print(f"verdicts (sequential)   : {tally('verdict_sequential')}")
    acc = [r for r in rows if r["verdict_sequential"] == "ACCEPT"]
    print(f"ACCEPT (sequential)     : {len(acc)}/{len(rows)} -> "
          f"{', '.join(r['factor'] for r in acc) or '(none)'}")
    if acc:
        fits = sorted(r["fitness"] for r in acc)
        print(f"  fitness of accepted   : {fits}")
        print(f"  median / best         : {fits[len(fits)//2]:+.3f} / {max(fits):+.3f}")
    print()
    print("gate-by-gate failures (independent):")
    counts: dict[str, int] = {}
    for r in rows:
        for g in r["failed_independent"]:
            counts[g] = counts.get(g, 0) + 1
    for g in sorted(counts):
        print(f"  {g}: {counts[g]}/{len(rows)} failed")
    print()
    print(f"thresholds in force: GATE_1>{config.GATE_1_MIN_SHARPE_VAL} "
          f"GATE_2>{config.GATE_2_MIN_DECAY_RATIO} "
          f"GATE_3>{config.GATE_3_MIN_PASS_RATE} "
          f"GATE_4<{config.GATE_4_TURNOVER_CAP} "
          f"GATE_5<{config.GATE_5_MAX_IC_VS_SOTA} "
          f"GATE_6<{config.GATE_6_MAX_NAN_RATIO}")
    print(f"split: train {config.TRAIN_SPLIT}  val {config.VAL_SPLIT}  "
          f"(test never touched, C2)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=None,
                   help="also write the rows as JSON")
    args = p.parse_args()
    setup_logging()

    rows = evaluate()
    print_table(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
