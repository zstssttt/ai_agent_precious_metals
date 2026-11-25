# pool_inventory.md — L0 inventory and spec-space coordinates

> The first half of Phase 0.5 (L0 → spec coordinates). The Track A/B fork
> decision is left to the second half of Phase 0.5.
> All numbers are actually computed by `python3 scripts/l0_report.py` and are
> reproducible (CLAUDE.md C4).
> Generated: 2026-11-21

---

## 1. Headline conclusions

| Question | Answer |
|---|---|
| Which 15 factors is L0 | see the §3 table; all have spec coordinates |
| Can the 15 factors be expressed in the spec space | **yes, 15/15**, and the 15 hashes are all distinct |
| How many does the v3 schema in `plan.md §3` express | **1/15** → the schema must go to v4, see §2 |
| Five-theme / five-template coverage | 5/5 · 5/5 |
| v4 structure-space size | **561 million** (plan.md §1 estimated "~tens of thousands", under by four orders of magnitude) |
| Usable test-segment indicators | **13/13**; the Phase -1 kill criterion does not trigger |

---

## 2. Why the schema must go from v3 to v4

The FactorSpec v3 in `plan.md §3` has this shape:

```json
{"template": "direction_x_intensity",
 "direction": {"indicator": "PMF0007", "op": "ma_diff"},
 "intensity": {"indicator": "PMF0042", "op": "abs_pct_change"}}
```

Two legs, one indicator per leg, 8 operators, no pre-transforms. After taking the
15 functions in `build_factors.py` apart line by line, **this schema can express
only `pmf_051056`**. There are four kinds of gap:

| Gap | L0 factors affected | Notes |
|---|---|---|
| **three-indicator risk_adjusted** | 128453 · 127111 · 125861 · 131810 | `corr(x,y) / risk` needs 3 indicators; v3 has only 2 slots |
| **pre-transforms** | all 15 | not one factor eats a raw indicator directly. `diff` / `pct_ret` / `lag1` / `abs` / `log_abs` / `rank` are necessities, not decoration |
| **missing operators** | 051113 · 104203 · 058341 · 102135 · 118259 · 055491 · 066409 · 011095 · 003678 | `corr_change` `corr_slope` `path_efficiency` `autocorr_abs` `boll_position` `ratio_over_logsize` `ma_spread_abs` are none of them in v3's 8-operator whitelist |
| **multiple windows within a leg** | 069545 · 118259 · 066409 · 051113 · 102135 | direction uses MA(20,60), intensity uses window 5. A single fast/slow global window cannot express this |

**This is exactly the kill criterion written in `plan.md` Phase 1** — "if 3
factors cannot be expressed as a spec → stop and redesign the schema". Hitting it
early in Phase 0 saved the two hours of building the builder and then tearing it
down.

### The four v4 changes

1. **Leg-based**: `legs: {role → leg}`, where role is set by the template
   (`risk_adjusted` has `numerator` + `risk`, `direction_x_intensity` has
   `direction` + `intensity`, etc.).
2. **Operators carry a kind type**: each operator declares an output type (`sign`
   / `intensity` / `persistence` / `position` / `signal` / `level` / `raw`), and
   the template says which kinds each role accepts. So "multiply two signs and
   still call it direction × intensity" is rejected at the schema level, not by
   prompt constraints.
3. **Pre-transform whitelist**: each input carries an ordered `pre` list, capped
   at 2 (AlphaAgent-style complexity control).
4. **Window-role binding**: a leg declares which role (`w1`/`w2`/`w3`) each window
   slot binds to. **The role name is structure; the value still belongs to the
   validator's grid** — A4 is not broken. Side benefit: `ma_cross` with both slots
   on the same role = identically zero, rejected by the schema, and the grid also
   auto-drops cells with `w_slow ≤ w_fast`.

The operator whitelist grows from 8 to **14**, all composable from the 8 ready-made
`ts_*` primitives in `build_factors.py` — no new math introduced, just promoting
the combinations the original author hand-wrote inside function bodies into named
operators.

---

## 3. L0's spec coordinates

`survivor` = the 6 with `selected=True` under `rolling_percentage` (the README's
headline methodology) in `results/portfolio_weights.csv`.
`grid` = the number of valid cells for that structure under the `{5,20,60}` grid;
`cx` = complexity (legs + window slots + pre-transforms).

| factor | survivor | theme | template | roles | orig_windows | grid | cx |
|---|---|---|---|---|---|---|---|
| pmf_128453 | yes | dollar | risk_adjusted | w1 | w1=5 | 3 | 10 |
| pmf_011095 | | dollar | identity | w1 | w1=5 | 3 | 9 |
| pmf_069545 | | dollar | direction_x_intensity | w1,w2,w3 | w1=5 w2=20 w3=60 | 9 | 6 |
| pmf_051113 | | goldprice | direction_x_intensity | w1,w2 | w1=5 w2=20 | 3 | 7 |
| pmf_127111 | | goldprice | risk_adjusted | w1 | w1=5 | 3 | 10 |
| pmf_125861 | yes | goldprice | risk_adjusted | w1 | w1=5 | 3 | 10 |
| pmf_131810 | yes | etf | risk_adjusted | w1 | w1=20 | 3 | 7 |
| pmf_104203 | | etf | signal_x_persistence | w1 | w1=5 | 3 | 6 |
| pmf_058341 | | etf | direction_x_intensity | w1 | w1=5 | 3 | 7 |
| pmf_102135 | | rates | signal_x_persistence | w1,w2 | w1=5 w2=20 | 3 | 5 |
| pmf_003678 | | rates | identity | w1 | w1=20 | 3 | 6 |
| pmf_118259 | | rates | direction_x_position | w1,w2,w3 | w1=5 w2=20 w3=60 | 9 | 5 |
| pmf_051056 | yes | risk | direction_x_intensity | w1,w2 | w1=5 w2=20 | 9 | 5 |
| pmf_055491 | yes | risk | direction_x_intensity | w1,w2 | w1=5 w2=20 | 9 | 7 |
| pmf_066409 | yes | risk | direction_x_intensity | w1,w2,w3 | w1=5 w2=20 w3=60 | 9 | 6 |

Structure expansion (verbatim output of `scripts/l0_report.py`):

```
pmf_128453  risk_adjusted[numerator=corr_change(diff+rank(PMF0012), diff+rank(PMF0036)) | risk=level(PMF0007)]
pmf_011095  identity[value=corr_slope(diff+rank(PMF0012), pct_ret+rank(PMF0031))]
pmf_069545  direction_x_intensity[direction=ma_cross(diff(PMF0012)) | intensity=abs_zscore(PMF0042)]
pmf_051113  direction_x_intensity[direction=ma_diff(PMF0007) | intensity=ma_spread_abs(pct_ret(PMF0020))]
pmf_127111  risk_adjusted[numerator=corr_change(diff+rank(PMF0011), diff+rank(PMF0031)) | risk=level(PMF0018)]
pmf_125861  risk_adjusted[numerator=corr_change(diff+rank(PMF0004), diff+rank(PMF0036)) | risk=level(PMF0007)]
pmf_131810  risk_adjusted[numerator=corr_change(diff(PMF0019), pct_ret+lag1(PMF0001)) | risk=level(PMF0007)]
pmf_104203  signal_x_persistence[persistence=path_efficiency(pct_ret(PMF0031)) | signal=aroon_balance(pct_ret(PMF0031))]
pmf_058341  direction_x_intensity[direction=ma_diff(PMF0031) | intensity=ratio_over_logsize(pct_ret+abs(PMF0013), PMF0031)]
pmf_102135  signal_x_persistence[persistence=autocorr_abs(PMF0042) | signal=ma_cross(PMF0042)]
pmf_003678  identity[value=corr_change(diff(PMF0002), diff+lag1(PMF0019))]
pmf_118259  direction_x_position[direction=ma_cross(PMF0042) | position=boll_position(PMF0042)]
pmf_051056  direction_x_intensity[direction=ma_diff(PMF0007) | intensity=abs_pct_change(PMF0042)]
pmf_055491  direction_x_intensity[direction=ma_diff(PMF0018) | intensity=ratio_over_logsize(diff+abs(PMF0042), PMF0036)]
pmf_066409  direction_x_intensity[direction=ma_cross(PMF0007) | intensity=ratio_over_logsize(abs(PMF0013), PMF0031)]
```

### What the coordinates tell us (all counts actually computed)

- **Template distribution is highly skewed**: `direction_x_intensity` 6 ·
  `risk_adjusted` 4 · `identity` 2 · `signal_x_persistence` 2 ·
  `direction_x_position` 1.
- **Indicator use is very uneven**: all 13 indicators are used (none idle), but
  frequency ranges 6× from 6 down to 1 —

  | high-freq | count | low-freq | count |
  |---|---:|---|---:|
  | PMF0007 GPR | 6 | PMF0020 onshore-offshore spread | 1 |
  | PMF0031 SPDR flow | 5 | PMF0011 Au9995 | 1 |
  | PMF0042 core CPI | 5 | PMF0004 COMEX | 1 |
  | PMF0012 DXY | 3 | PMF0001 UST10Y | 1 |
  | PMF0036 SPDR cash flow | 3 | PMF0002 TIPS | 1 |

  **L2's real blank space is "low-frequency indicators × common operators"**: the
  three indicators of the whole goldprice theme appear only 3 times combined, and
  TIPS — gold's textbook driver — made it into just 1 factor.

- **Cross-theme pairing is not blank but the norm: 12/15 factors cross themes**, 5
  of them spanning 3 themes. So `plan.md §1` listing "cross-theme combinations like
  DXY × SPDR flow" as L2's unexplored area **is wrong** — `pmf_128453` is exactly
  that combination. L2's blank space is in "low-frequency indicators" and "operator
  substitution", not in cross-theme.
- **A theme label is narrative, not structure**: `pmf_055491` is labeled risk, but
  it eats indicators from etf + rates + risk. The theme label should be read as
  "which story this factor tells", not "which kind of data it uses". This must go
  into the Synthesis system prompt, or the agent will wrongly assume the theme
  restricts the choice of indicators.
- **Window choice is extremely uniform**: the 15 factors' original windows use only
  the three values `{5, 20, 60}`, and 10/15 use a single window role. In other
  words, **humans never checked whether these structures still hold under other
  windows** — so the window-grid scan carries real information, it is not a
  formality.

---

## 4. Structure-space size (input to the Track A/B decision)

Actually computed by `python3 scripts/l0_report.py --space`:

| Config | inputs per slot | total structures |
|---|---:|---:|
| full v4 (pre-transforms on) | 481 | **561,284,191** |
| pre-transforms off (indicator + operator only) | 13 | **14,599** |

By template (pre-transforms off): `risk_adjusted` 7,267 · `direction_x_intensity`
5,408 · `signal_x_persistence` 1,014 · `identity` 572 · `direction_x_position` 338.

**Impact on the narrative**: plan.md §1's "~tens of thousands" estimate was under
by four orders of magnitude. This makes the claim "exhaustion is infeasible, so
sample efficiency is the real problem" **stronger**, but it also means:

> Track B's `exhaustive_sweep.py` must never scan the whole space. A feasible L1
> annotation space is "pre-transforms off + limited to 2 templates" ≈ 12,675
> structures × 3–9 cells ≈ 40k–110k backtests, still too large. **Recommend
> further restricting to a single template + an indicator-pair subset, landing on
> the 500–2000 cells plan.md mentions.**

The specific restriction scheme is left to the second half of Phase 0.5, out of
scope for this round.

---

## 5. Two data-side findings (affecting config, already landed)

1. **ffill fake tail**: 8 of the 13 indicators' CSVs are written all the way to
   `2026-12-31`, but the last 265 rows are entirely unchanged padding values. Real
   observations uniformly stop at **2025-11-28**. `config.DATA_CUTOFF` hard-truncates
   this, or the backtest would run on a whole year of fake data.
2. **Test segment is usable**: all 13 indicators cover through 2025-11-28, while the
   original project's in-sample ends 2023-12-31. So the split is set to
   - train `2015-01-01 → 2021-12-31`
   - val `2022-01-01 → 2023-12-31`
   - test `2024-01-01 → 2025-11-28` (locked by C2)

   This is a **genuinely out-of-time** test segment, and needs no akshare data
   backfill. The Phase -1 kill criterion (fewer than 5 extendable indicators) does
   **not** trigger.

> Also: `补齐后数据/` actually holds 42 series, far more than the 13 used by the 15
> factors (VIX, gold/silver ratio, Fed total assets, COMEX speculative net longs,
> US M2, Michigan consumer expectations, SOFR, 30Y breakeven, gold-ETF volatility
> index…). Expanding the whitelist would significantly enlarge L2, but each needs a
> theme assignment + coverage check first. Recorded in `BACKLOG.md`.
