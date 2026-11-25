"""① Specification Unit — all constants live here. Nothing hardcodes a path.

CLAUDE.md C2: ``split`` is only ever "train" or "val" downstream. The test
window is declared here so that ``scripts/unlock_test.py`` can find it, but no
module other than that script may read ``TEST_SPLIT``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ======================================================================
# Paths
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Aligned indicator lake. ``data/aligned`` is a symlink to the project-local
# ``补齐后数据/`` folder, so nothing depends on an external drive (CLAUDE.md
# code conventions). Override for a different mount with GOLD_ALIGNED_ROOT.
ALIGNED_ROOT = Path(os.environ.get("GOLD_ALIGNED_ROOT", PROJECT_ROOT / "data" / "aligned"))

# The read-only upstream repo imported under C3 (never modified, only imported).
LEGACY_SRC_ROOT = PROJECT_ROOT / "final_src_github"
LEGACY_RESULTS = LEGACY_SRC_ROOT / "results"

JOURNAL_DIR = PROJECT_ROOT / "journal"
TRIALS_PATH = JOURNAL_DIR / "trials.jsonl"
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
RULES_DIR = KNOWLEDGE_DIR / "rules"
FINDINGS_DIR = KNOWLEDGE_DIR / "findings"
FAILURES_DIR = KNOWLEDGE_DIR / "failures"
DOCS_DIR = PROJECT_ROOT / "docs"


# ======================================================================
# Time splits
# ======================================================================
# Evidence (scan run 2026-11-21 over data/aligned): every one of the 13 PMF
# indicators carries genuine observations through 2025-11-28. Rows dated after
# that are forward-fill padding emitted by the alignment script (265 identical
# trailing rows out to 2026-12-31 on 8 of the 13 files) and MUST be truncated.
DATA_CUTOFF = "2025-11-28"

# The upstream repo's published in-sample period ends 2023-12-31, so a test
# window of 2024-01-01 → DATA_CUTOFF is genuinely out-of-time relative to every
# number in final_src_github/README.md.
TRAIN_SPLIT = ("2015-01-01", "2021-12-31")
VAL_SPLIT = ("2022-01-01", "2023-12-31")

# C2: LOCKED. Reachable only through scripts/unlock_test.py with the explicit
# --i-understand-this-is-one-shot flag. Importing this name anywhere else is a bug.
TEST_SPLIT = ("2024-01-01", DATA_CUTOFF)

SPLITS = {"train": TRAIN_SPLIT, "val": VAL_SPLIT}
ALLOWED_SPLITS = ("train", "val")


# ======================================================================
# Window grid (CLAUDE.md A4: windows are NOT part of a spec)
# ======================================================================
# A spec never names a window *value*. It names window *roles* — which operator
# slots share one window and which move independently. That is structure. The
# validator assigns values from WINDOW_VALUES to the roles and scans the grid.
#
# Why roles rather than one global (fast, slow) pair: L0's pmf_069545 takes its
# direction from a 20/60 MA cross and its intensity from a 5-day z-score, so a
# single spec genuinely needs three independent windows.
WINDOW_VALUES = (5, 20, 60)
ROLE_NAMES = ("w1", "w2", "w3")
MAX_WINDOW_ROLES = 3
# Hard ceiling on grid size, so one spec can never dominate the trial budget.
# 3 values ** 3 roles = 27, matching plan.md §3.
MAX_GRID_CELLS = 27


# ======================================================================
# The six gates (plan.md §4 L1). Provisional — Phase 3 recalibrates them
# against the 15 L0 factors and records the evidence in docs/calibration.md.
# ======================================================================
GATE_1_MIN_SHARPE_VAL = 0.0          # sharpe_median_val must exceed this
GATE_2_MIN_DECAY_RATIO = 0.5         # sharpe_median_val / sharpe_median_train
GATE_3_MIN_PASS_RATE = 0.3           # share of grid cells clearing the gates
GATE_4_TURNOVER_CAP = 25.0           # annualised turnover
GATE_5_MAX_IC_VS_SOTA = 0.70         # signal-level dedup vs the SOTA set
GATE_6_MAX_NAN_RATIO = 0.3           # share of NaN in the factor series

# GATE_3 (calibrated 2026-11-23, docs/calibration.md §3.3): lowered 0.4 -> 0.3.
# 9 of the 15 L0 factors sit at pass_rate exactly 0.333, so 0.3 and 0.4 are not
# neighbouring settings — they are opposite sides of a cliff that flips 60% of
# the sample at once. At 0.3 the gate passes 12/15 and the SOTA set reaches 6,
# which is what Phase 4 needs for GATE_5 to mean anything; at 0.4 it passes 3.
# KNOWN COST: for a 1-role spec the grid is only 3 cells, so 0.3 lets "1 of 3
# windows worked" count as robust. That is weak evidence and must be stated in
# the report. BACKLOG B9 tracks the real fix (resolution-aware thresholds).

# GATE_4 evidence: the 45 rows of final_src_github/results/single_factor_metrics.csv
# span annualised turnover 7.77 – 20.67 (median 15.77). A cap of 25 sits above
# every hand-designed factor while still rejecting pathological churn.

# GATE_5 (calibrated 2026-11-23, docs/calibration.md §3.5): tightened 0.8 -> 0.70.
# All 105 pairwise |Spearman IC| among the L0 factors: median 0.033, p90 0.117,
# then a jump to 0.723 and 0.796. The old 0.8 caught neither of the two obvious
# near-duplicates — one of which shares a character-identical direction leg.
# Any threshold in (0.42, 0.72] catches exactly those two and no others; 0.70
# sits at the conservative edge of that gap.


# ======================================================================
# Search budget
# ======================================================================
MAX_TRIALS = 40                      # plan.md §8 suggests 30–50; provisional
MAX_TOOL_CALLS_PER_TRIAL = 8
MAX_INNER_RETRIES = 3                # schema-validation feedback loop (A/E of the patterns note)
MAX_PRE_TRANSFORMS_PER_INPUT = 2     # complexity control (AlphaAgent §1.5)
SCAN_TIMEOUT_SECONDS = 30            # one spec's grid scan; longer means a bug, not big data

RANDOM_SEED = 20261121


# ======================================================================
# ② Synthesis Unit — the only place an LLM is allowed (C1)
# ======================================================================
# Everything here is an environment override so switching providers never
# requires a code edit. Defaults target DeepSeek: it is reachable from the
# mainland without a proxy, which matters more than price over a 25-60 minute
# loop — a dropped proxy mid-run is the realistic failure mode, not the bill.
#
# Verified against api-docs.deepseek.com on 2026-11-23:
#   base URL        https://api.deepseek.com        (OpenAI-compatible)
#   strict mode     https://api.deepseek.com/beta   (required for strict:true)
#   models          deepseek-flash | deepseek-v4-pro
#   key env var     DEEPSEEK_API_KEY
LLM_PROVIDER = os.environ.get("HARNESS_LLM_PROVIDER", "deepseek")
LLM_MODEL = os.environ.get("HARNESS_LLM_MODEL", "deepseek-v4-pro")
LLM_BASE_URL = os.environ.get("HARNESS_LLM_BASE_URL", "")   # "" = provider default
LLM_API_KEY_ENV = os.environ.get("HARNESS_LLM_API_KEY_ENV", "")  # "" = provider default

# DeepSeek strict mode guarantees the tool arguments match the JSON schema, so
# the inner retry loop almost never fires. Turning it off falls back to plain
# function calling plus our own validation.
LLM_STRICT_TOOLS = os.environ.get("HARNESS_LLM_STRICT", "1") not in ("0", "false", "")

# Thinking costs output tokens. "high" for the synthesis call (proposing an
# economically motivated structure is the one genuinely hard judgement in the
# loop); the alignment judge runs cheaper — see LLM_JUDGE_*.
LLM_REASONING_EFFORT = os.environ.get("HARNESS_LLM_EFFORT", "high")
LLM_THINKING = os.environ.get("HARNESS_LLM_THINKING", "enabled")

LLM_JUDGE_MODEL = os.environ.get("HARNESS_LLM_JUDGE_MODEL", "")  # "" = same as LLM_MODEL
LLM_JUDGE_EFFORT = os.environ.get("HARNESS_LLM_JUDGE_EFFORT", "low")

LLM_MAX_RETRIES = 4          # transport-level retries on 429 / 5xx / timeout
LLM_TIMEOUT_SECONDS = 180
# Shared ceiling for reasoning + answer tokens, NOT a per-answer budget. With
# LLM_REASONING_EFFORT="high" the synthesis call can spend ~4k tokens thinking
# before it emits a single tool call, so 4096 let reasoning eat the whole budget
# and the turn came back finish_reason="length" with zero usable output — a
# silent money leak (HTTP 200, no exception), measured on the 2026-11-24 smoke
# run as 2 of 3 trials wasting their first call. 8192 gives thinking room to
# finish AND still answer. Raising the ceiling does not raise the bill: you pay
# only for tokens actually generated, and this removes the wasted re-ask.
LLM_MAX_OUTPUT_TOKENS = 16384

# Every request and response lands here, whole. The journal records decisions;
# this records what the model was actually shown (plan.md Phase 4: full trace on disk).
TRACE_PATH = JOURNAL_DIR / "llm_trace.jsonl"

# Hypothesis-alignment gate (plan.md Phase 4 item 4). A spec whose stated
# economics do not match its structure is rejected before the backtest, which
# is what stops the agent inventing a story after the fact.
ALIGNMENT_MIN_SCORE = 3      # 1-5 scale; below this the trial is MISALIGNED

# Stop early when the agent is going in circles rather than burning the budget
# (plan.md Phase 4 kill criterion).
MAX_CONSECUTIVE_DUPLICATES = 5


# ======================================================================
# Backtest conventions — mirrored from the read-only position_mapping.py (C3).
# Kept here so harness code never hardcodes them; the wrapper asserts they match.
# ======================================================================
COST_RATE = 0.0003
HOLDING_PERIOD = 10
SMOOTH_WINDOW = 5
TRADING_DAYS_PER_YEAR = 252

# The traded asset: Au9995 daily close. The upstream default path points inside
# final_src_github/, which does not carry the data; the real file lives in the
# aligned lake.
AU9995_PATH = ALIGNED_ROOT / "补齐后日频数据2.0" / "收盘价_Au9995_S0035818.csv"

# Of the three mappings upstream exposes, rolling_percentage is the one the
# published portfolio actually selects on, so the harness scans that one only.
# Scanning all three would triple the grid and amount to searching over the
# execution layer, which is not what the agent is supposed to be searching.
POSITION_TYPE = "rolling_percentage"
POSITION_ROLLING_WINDOW = 252        # mirrors position_mapping.ROLLING_WINDOW
POSITION_MIN_PERIODS = 60            # mirrors position_mapping.MIN_PERIODS_ROLLING

# Published in-sample reference run (results/single_factor_metrics.csv) used by
# tests/test_phase2.py to prove the wrapper reproduces upstream bit-for-bit.
# A7: this is an in-sample number and must never be a comparison target.
LEGACY_CUTOFF_DATE = "2023-12-31"


# ======================================================================
# Cell-level pass predicate — what makes one grid cell "pass" (plan.md §3)
# ======================================================================
# plan.md writes pass_rate = mean(passes_gates(cell)), but GATE_3 *is*
# pass_rate > 0.4, so feeding all six gates back in is circular. Only the gates
# that are decidable from a single cell in isolation may participate:
#
#   GATE_1 val Sharpe > 0     cell-local      -> in
#   GATE_2 decay ratio > 0.5  cell-local      -> in
#   GATE_4 turnover < cap     cell-local      -> in
#   GATE_3 pass_rate > 0.4    IS pass_rate    -> out (circular)
#   GATE_5 IC vs SOTA         needs SOTA set  -> out (factor-level, Phase 3)
#   GATE_6 nan_ratio < 0.3    cell-local but  -> out (factor-level: a spec whose
#                                                 every cell is empty should be
#                                                 rejected outright, not scored
#                                                 as pass_rate = 0)
#
# So pass_rate measures "in how many corners of the window grid does this
# structure make money out of sample without excessive churn".
CELL_PASS_GATES = ("GATE_1", "GATE_2", "GATE_4")


# ======================================================================
# Logging
# ======================================================================
LOG_LEVEL = os.environ.get("HARNESS_LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)-18s %(message)s"
