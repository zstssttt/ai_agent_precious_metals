"""Print the L0 inventory table and regenerate docs/pool_inventory.md's data section.

Usage:
    python3 scripts/l0_report.py            # print to stdout
    python3 scripts/l0_report.py --markdown # emit the markdown table
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import config, setup_logging  # noqa: E402
from harness.l0_catalog import L0_SPECS, L0_SURVIVORS, L0_WINDOWS  # noqa: E402
from harness.spec import INDICATORS, OPERATORS, PRE_TRANSFORMS, TEMPLATES, window_grid  # noqa: E402


def rows() -> list[dict[str, str]]:
    out = []
    for name, spec in L0_SPECS.items():
        windows = L0_WINDOWS[name]
        out.append({
            "factor": name,
            "survivor": "yes" if name in L0_SURVIVORS else "",
            "theme": spec.theme,
            "template": spec.template,
            "structure": spec.structure_label,
            "indicators": " ".join(spec.indicators),
            "roles": ",".join(spec.window_roles),
            "orig_windows": " ".join(f"{k}={v}" for k, v in sorted(windows.items())),
            "grid": str(spec.n_grid_cells),
            "cx": str(spec.complexity),
            "hash": spec.spec_hash[:10],
        })
    return out


def print_plain() -> None:
    data = rows()
    headers = ["factor", "survivor", "theme", "template", "roles", "orig_windows",
               "grid", "cx", "hash"]
    widths = {h: max(len(h), max(len(r[h]) for r in data)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(line)
    print("-" * len(line))
    for r in data:
        print("  ".join(r[h].ljust(widths[h]) for h in headers))
    print()
    for r in data:
        print(f"{r['factor']}  {r['structure']}")
    print()
    print(f"L0 size                : {len(L0_SPECS)}")
    print(f"survivors              : {len(L0_SURVIVORS)} -> {', '.join(L0_SURVIVORS)}")
    print(f"distinct structures    : {len({s.spec_hash for s in L0_SPECS.values()})}")
    print(f"themes covered         : {len({s.theme for s in L0_SPECS.values()})}/5")
    print(f"templates covered      : {len({s.template for s in L0_SPECS.values()})}/{len(TEMPLATES)}")
    print(f"indicators whitelisted : {len(INDICATORS)}")
    print(f"operators whitelisted  : {len(OPERATORS)}")
    print(f"total L0 grid cells    : {sum(s.n_grid_cells for s in L0_SPECS.values())}")
    print(f"window grid values     : {config.WINDOW_VALUES}")
    print(f"splits                 : train {config.TRAIN_SPLIT}  val {config.VAL_SPLIT}")


def print_markdown() -> None:
    data = rows()
    headers = ["factor", "survivor", "theme", "template", "structure",
               "roles", "orig_windows", "grid", "cx"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in data:
        cells = [r[h].replace("|", "\\|") for h in headers]
        cells[4] = f"`{cells[4]}`"
        print("| " + " | ".join(cells) + " |")


def _n_pre_sequences(max_len: int, n_pre: int) -> int:
    """Ordered pre-transform sequences of length 0..max_len, no repeats."""
    total, falling = 0, 1
    for k in range(max_len + 1):
        if k:
            falling *= (n_pre - k + 1)
        total += falling
    return total


def print_space() -> None:
    """Size the v4 structure space — the input to plan.md §1's Track A/B call."""
    n_ind = len(INDICATORS)
    n_pre = len(PRE_TRANSFORMS)
    seqs_full = _n_pre_sequences(config.MAX_PRE_TRANSFORMS_PER_INPUT, n_pre)
    seqs_none = 1

    for label, seqs in (("with pre-transforms", seqs_full), ("pre-transforms off", seqs_none)):
        slot = n_ind * seqs           # distinct (indicator, pre) inputs
        total = 0
        print(f"\n{label}: {slot} distinct inputs per slot")
        for template, roles in sorted(TEMPLATES.items()):
            per_template = 1
            for _role, kinds in sorted(roles.items()):
                role_total = sum(
                    slot ** op.arity
                    for op in OPERATORS.values() if op.kind in kinds
                )
                per_template *= role_total
            total += per_template
            print(f"  {template:<22} {per_template:>15,}")
        print(f"  {'TOTAL':<22} {total:>15,}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--space", action="store_true",
                        help="size the structure space instead of listing L0")
    args = parser.parse_args()
    setup_logging()
    if args.space:
        print_space()
    elif args.markdown:
        print_markdown()
    else:
        print_plain()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
