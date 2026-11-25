"""Gold Factor Research Harness — an agentic factor-research loop.

Unit map (plan.md §2):

    config.py   ① Specification — splits, gates, window grid, budget
    spec.py     ① FactorSpec schema, whitelists, canonical hash
    l0_catalog.py  the 15 hand-designed factors as v4 specs (human baseline)
    journal.py  memory — trials.jsonl append + structured queries
    builder.py  ③ Implementation (Phase 1)
    validator.py ④ Validation (Phase 2)
    analyst.py  ⑤ Analysis (Phase 3)
    tools.py / loop.py  ② Synthesis (Phase 4)
"""

import logging

from harness import config

__version__ = "0.1.0"


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, per the CLAUDE.md code conventions."""
    logging.basicConfig(
        level=getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO),
        format=config.LOG_FORMAT,
    )


__all__ = ["config", "setup_logging", "__version__"]
