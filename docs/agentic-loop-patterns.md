# Agentic factor-research loop — pattern study notes

Goal: understand **how an agentic flow actually iterates**, and what happens
inside one iteration in the specific domain of "factor research".

---

## Part 1: general patterns — what an agentic research loop looks like

### 1.1 Core skeleton: split the loop into "units", don't write one giant prompt

This is the first consensus across all mature frameworks. Microsoft RD-Agent(Q)
<cite index="27-1">splits quant research into five tightly-coupled units: Specification (scenario definition), Synthesis (idea generation), Implementation (code development), Validation (backtest), and Analysis (result evaluation and task scheduling), forming a closed loop under unified input/output constraints that mimics a human quant researcher's trial-and-error process.</cite>

Mapped onto two stages: <cite index="26-1">the Research stage dynamically generates goal-aligned prompts, forms hypotheses from domain priors, and maps them to concrete tasks; the Development stage has a code-generation agent implement the code and backtest it on a real market; the two stages are connected by a feedback stage that evaluates results and guides the next iteration.</cite>

**Why split into units**: each unit's input/output is contractual, so it can be
tested, replaced, and time-limited independently. A single prompt doing everything
never tells you which step went wrong.

> **Your project's mapping**: Phase 1's builder = Implementation, Phase 0's
> evaluator = Validation, Phase 4's judge = Analysis. Your Synthesis is the agent
> itself.

---

### 1.2 Two views: local view vs global view

This is the design I think is most worth learning. In RD-Agent(Q), <cite index="27-1">the Analysis Unit works from the local view of "this experiment", while the Synthesis Unit keeps a global view across the entire experiment history. Their interaction balances short-term responsiveness against long-term exploration.</cite>

**In plain terms**:

- **Local view** (the reviewer): looks only at this trial's numbers and judges
  pass/fail. It does not care about history.
- **Global view** (the researcher): looks at all past trials and decides which
  direction to go next. It does not fuss over single-trial detail.

With only a local view, the agent falls into local search — "always just tweak the
last factor's parameters". With only a global view, it ignores the concrete
diagnosis it just received.

> **Your project's mapping**: `judge.py` is the local view (the 4 gates see only
> the current spec); the `knowledge/` + `trials.jsonl` fed to the agent is the
> global view. **Do not mix the two into one prompt.**

---

### 1.3 Feedback must be a structured diagnosis, not a single score

This is where beginners most often go wrong. Returning only `Sharpe = 0.3` tells
the agent "you were wrong" but not where, so it can only guess blindly.

RD-Agent(Q)'s approach: <cite index="27-1">the Analysis Unit scores the hypothesis, task, and experiment results on multiple dimensions each round; if the experiment beats SOTA under the current action type, its result is added to the corresponding SOTA set. The unit then diagnoses the failed strategy and generates targeted improvement suggestions, which are fed back to the Synthesis Unit to guide the next hypotheses.</cite>

More interesting is a **controlled experiment**. One repo specifically tested
"whether feedback richness actually helps", <cite index="20-1">designing three prompt groups: group A is pure symbolic search, given performance feedback only, no financial vocabulary; group B = A plus price-volume theory priors and a short hypothesis per formula; group C = B plus rule-based diagnostic feedback (rules R1–R3, translated into repair instructions). Two LLMs and a genetic-programming baseline were compared under the same evaluation budget, all methods sharing the same operator library, the same fitness function, and the same evaluator.</cite>

**Note group C's key phrase: translate the diagnostic rules "into repair
instructions".** Not "your factor's IC is too low", but "your factor fails in the
low-volatility regime; change the volatility condition from a threshold test to a
continuous weighting".

> **Your project's mapping**: in the plan, judge's `REFINE` state must carry a
> structured diagnosis. My example was `"train 0.9 / val 0.1, decay ratio 0.11,
> suspected overfitting, suggest widening the window rather than tightening it"` —
> that is "translate into repair instructions".

---

### 1.4 Dedup is a hard requirement, not an optimization

Both frameworks put it up front in the flow:

- RD-Agent(Q): <cite index="25-1">factors are deduped first, filtering redundant signals by computing their correlation against the existing SOTA factor library.</cite>
- alpha-agent: <cite index="13-1">accepted factors are persisted to SQLite with structured AST dedup.</cite>

Note these are **two different kinds of dedup**, both needed:

| Type | The question it asks | Implementation |
|---|---|---|
| **Structural dedup** | Have I written this expression before? | AST compare / spec hash |
| **Signal dedup** | How correlated is this factor's values with existing factors? | compute correlation |

Two factors that differ structurally but correlate at 0.95 are wasted effort.

> **Your project's mapping**: the plan's `check_if_tried` only does structural
> dedup (spec hash). **Signal dedup is a 20-line feature worth adding**: compute
> the correlation of the new factor series against already-ACCEPTed factors, and
> reject above 0.8. This is exactly what RD-Agent does, costs very little, and is a
> clear interview plus. I'd suggest adding it to Phase 4.

---

### 1.5 The regularization trio: the general prescription against an agent gaming the score

AlphaAgent (KDD'25) defines the problem most clearly. It names two causes directly:
<cite index="37-1">one is overfitting from excessive data mining ("p-hacking"), producing pseudo-factors significant in backtest but decaying quickly live; the other is factor crowding, too many investors adopting similar strategies.</cite>

Three corresponding mechanisms: <cite index="36-1">(1) complexity control via the symbolic expression tree and parameter count; (2) hypothesis alignment using an LLM to assess the semantic consistency between the market hypothesis and the generated factor; (3) a novelty constraint via AST similarity against an existing factor library (e.g. Alpha101). AlphaAgent formalizes factor construction with an operator library and an abstract syntax tree, quantifies factor originality via pairwise subtree-isomorphism detection, and uses</cite> an LLM to score the consistency among hypothesis, description, and expression.

Breaking the three down, each guards against:

- **Complexity control** → stops the agent from using pure-fitting forms like
  `** 0.5` or triple nesting. **Parameter counting** is the cheapest implementation.
- **Hypothesis alignment** → stops the agent from inventing an economic story
  after the fact. It must write the hypothesis **first**, and the expression must be
  semantically consistent with it.
- **Originality / novelty** → stops the agent from repeatedly producing the same
  class of factor.

> **Your project's mapping**: you already have (1) (discrete parameters + operator
> whitelist) and (3) (spec hash). (2) is **free for you but not yet written into
> acceptance** — the `hypothesis` field in the spec schema lets a separate LLM call
> score "hypothesis vs expression" consistency and reject on mismatch. This is the
> standard **LLM-as-judge** usage, and the most direct realization of that phrase on
> your resume.

---

### 1.6 Inner and outer loops

The same "iteration" actually has two layers — don't conflate them:

A Vietnamese team's work separates this most clearly: <cite index="15-1">the system uses an inner loop to raise code quality before the backtest, and an outer loop to evaluate financial performance, absorb feedback, and accumulate the best qualifying factors into a knowledge base.</cite>

- **Inner loop**: code won't run → fix → re-run. The goal is **executability**.
- **Outer loop**: code runs but Sharpe is bad → change hypothesis → retry. The goal
  is **performance**.

RD-Agent caps the inner loop hard: <cite index="27-1">Co-STEER's inner optimization loop is capped at 10 iterations per task for both factor and model workflows; the Implementation Unit is limited to 600 seconds per task, the Validation Unit to 3600 seconds.</cite>

> **Your project's mapping**: you chose the spec approach, which **nearly
> eliminates the inner loop** (no code to fix, only parameters to validate). This is
> the fundamental reason the spec approach saves time. You only need the outer loop.

---

### 1.7 Choosing the next direction: three strategies

This is the most central question of "iteration" — **how does the agent decide what
the next candidate is**?

| Strategy | How | Representative |
|---|---|---|
| **Bandit scheduling** | treat "change factor / change model / switch theme" as arms of a multi-armed bandit, chosen adaptively by historical payoff | RD-Agent(Q): <cite index="28-1">the Analysis Unit uses a multi-armed-bandit scheduler to adaptively pick the next optimization direction</cite> |
| **Directed mutation** | predefine a set of mutation operators (widen the window, swap operator, swap indicator…) and pick one by rule or score | QuantGPT: <cite index="12-1">8 directed mutation strategies + high-score factor crossover + an adaptive strategy selector</cite> |
| **LLM decides freely** | give the LLM the history and diagnosis and let it think up the next step | alpha-agent / AlphaAgent |

**Which to pick for 15 hours**: the third. The first two require you to first
define the mutation-operator space or the arm set — extra engineering; and "the LLM
decides from the diagnosis" is itself the agentic selling point. Bandit can go into
`BACKLOG.md` as a next step.

---

### 1.8 Budget, SOTA set, knowledge accumulation

Three closing mechanisms:

- **SOTA set**: <cite index="27-1">if an experiment is judged to beat SOTA under the current action type, its result is added to the corresponding SOTA set</cite>; RD-Agent also maintains a <cite index="22-1">"knowledge forest" holding the hypotheses, implementations, and results of all past experiments, adaptively adjusting hypothesis complexity by recent performance.</cite>
- **Cross-session knowledge accumulation**: this is QuantGPT's three-bucket
  structure — <cite index="16-1">rules are verified stable rules (must obey), findings are empirical findings (reference), failures are disproved paths (do not repeat); the knowledge base accumulates across sessions, so the 10th research session can directly use all findings of the prior 9, avoiding repeat experiments, obeying verified rules, and routing around disproved paths. This is not chat history, it is a structured research asset.</cite>
- **Hard time/count budget**: <cite index="24-1">RD-Agent runs in a sandbox, with historical data read-only, a per-backtest execution timeout (e.g. 10 minutes) to prevent runaway loops, CPU/memory caps, and an audit log of every factor expression and backtest parameter.</cite>

> **"This is not chat history, it is a structured research asset"** — I'd suggest
> absorbing this line straight into your interview talking points. It states, in one
> sentence, the real role of RAG in an agent system.

---

## Part 2: what actually happens inside one iteration

Stringing the above patterns into an executable flow. This is what you should have
in mind when you have Claude Code implement `loop.py`:

```
【Round N start】
  ↓
① Read global state
   - knowledge/rules/     → what must be obeyed
   - knowledge/failures/  → which paths are already disproved
   - SOTA set             → the current best spec
   - last round's structured diagnosis → why it was rejected, how to change it
  ↓
② Form a hypothesis (Synthesis, global view)
   agent outputs: an economic hypothesis (natural language) + what to test this time
   ↓ constraint: the hypothesis must land in one of the five themes
  ↓
③ Produce a candidate spec (Implementation)
   agent outputs: JSON spec + parent_spec_hash + mutation_note
   ↓ validate: parameters in the discrete set? operators in the whitelist?
   ↓ fails → feed the error back, return to ③ (this is the inner loop, cap 3)
  ↓
④ Pre-filter (dedup, before paying for a backtest)
   - structural dedup: spec hash seen before?
   - complexity: parameter count over cap?
   - hypothesis alignment: is the hypothesis semantically consistent with the spec? (LLM-as-judge)
   ↓ any fails → record to journal, jump to ⑦
  ↓
⑤ Validation (deterministic, no LLM)
   builder(spec) → series
   evaluate(series, "train") + evaluate(series, "val")
   ↓ note: the test segment is physically unreachable
  ↓
⑥ Adjudicate (Analysis, local view)
   4 hard gates → fitness = min(sharpe_train, sharpe_val)
   signal dedup: correlation with already-ACCEPTed factors > 0.8?
   ↓
   ACCEPT → join the SOTA set
   REJECT → write knowledge/failures/, with the reason
   REFINE → generate a "repair instruction" diagnosis, carry into the next round
  ↓
⑦ Full trace
   append one line to trials.jsonl (success or failure)
  ↓
【Check budget: N >= MAX_TRIALS ? stop : round N+1】
```

**Three easily-missed details**:

1. **Dedup before the backtest** (step ④ before ⑤). The backtest is the slowest
   step in the whole loop; deduping after it wastes time.
2. **REJECT must also leave a trace.** A failure record is input to the next round,
   not garbage.
3. **The diagnosis must be carried into the next round, not just written to the
   log.** The log is for humans, the diagnosis is for the agent — two different
   things.

---

## Part 3: reading list (by priority)

### Must-read ①: AlphaAgent — understand "why regularize"
- Paper: `arxiv.org/abs/2502.16789` (KDD '25)
- Code: `github.com/RndmVariableQ/AlphaAgent`
- **What to read**: §1's breakdown of alpha decay's two causes (p-hacking + factor
  crowding) and the formalization of the three regularization mechanisms.
- **Why it matters most to you**: it directly answers your "how to iterate on
  Sharpe" question. Your judge design is a simplified version of this paper.

### Must-read ②: RD-Agent(Q) — understand "how to split the loop into units"
- Paper: `arxiv.org/abs/2505.15155` (NeurIPS 2025)
- Code: `github.com/microsoft/RD-Agent`
- **What to read**: the five-unit decomposition figure (Fig. 2), the local/global
  division of labor between the Analysis Unit and the Synthesis Unit, and the
  timeout and iteration-cap settings in the appendix.
- **Don't touch the code first**: this repo is huge — <cite index="24-1">Qlib has 48k stars and is a production-grade platform</cite> — diving in will get you lost. Read the paper's architecture figure first.
- **A reference point**: <cite index="23-1">experiments show that under a cost of under $10, RD-Agent(Q)'s annualized return is about 2× a benchmark factor library while using over 70% fewer factors.</cite> The cost magnitude is useful for estimating your `MAX_TRIALS`.

### Must-read ③: Alpha-Mining-Agent — understand "the controlled experiment on feedback richness"
- Code: `github.com/silinzhang66/Alpha-Mining-Agent`
- **What to read**: the design of the A/B/C three prompt groups, how rules R1–R3 are
  translated into repair instructions, and its experimental discipline — <cite index="20-1">the design is frozen before the main experiment starts (docs/design_freeze.md), and each of the two test periods is evaluated only once.</cite>
- **Why**: this is the only one that treats "does diagnostic feedback actually help"
  as a controlled variable. And its **test lock** discipline is exactly what your
  Phase 6 should copy.

### Optional ④: QuantGPT — see the engineering of the three-bucket knowledge base
- Code: `github.com/Miasyster/QuantGPT`
- **What to read**: the directory organization of `research_notes/knowledge/`,
  `iteration.py`, `anti_overfit.py` (<cite index="12-1">4 statistical anti-overfitting tests</cite>), `rolling_validator.py`.
- Note it is <cite index="12-1">driven by Claude via MCP, with 15 tools forming the agent's toolbox</cite> — the tool design is directly referenceable.

### Optional ⑤: alpha-agent — see the minimal readable four-agent implementation
- Code: `github.com/zzzhhn/alpha-agent`
- **What to read**: the responsibility split of the four agents under `agents/`, and
  how `pipeline/state.py`'s <cite index="13-1">immutable PipelineState flows through each agent</cite>.
- **Why this one**: small-scale, readable, <cite index="13-1">the feedback loop runs at most 3 rounds before accept/reject/escalate</cite>, the closest reference to your MVP scale.

### Index: `github.com/Tom-roujiang/Awesome-LLM-Quantitative-Trading-Papers`
A master index of papers for when you want to dig deeper.

---

## Part 4: which of these patterns to copy in your 15 hours

| Pattern | Copy? | Reason |
|---|---|---|
| Unit split (5 units) | **copy** | already in plan.md, the basis of the Phase split |
| local / global view separation | **copy** | zero cost, just prompt organization |
| structured diagnosis (repair instructions) | **copy** | judge's REFINE must carry a diagnosis |
| structural dedup (spec hash) | **copy** | already in the plan |
| **signal dedup (correlation)** | **add it** | 20 lines, same as RD-Agent, the highest-ROI addition |
| **hypothesis alignment (LLM-as-judge)** | **add it** | the spec already has a hypothesis field; one LLM call |
| complexity control | **copy** | discrete parameters + whitelist already implement it |
| three-bucket knowledge | **copy** | already in the plan |
| inner loop (code repair) | **don't** | the spec approach eliminated it |
| Bandit scheduling | **don't** | goes to BACKLOG |
| genetic crossover / mutation engine | **don't** | goes to BACKLOG |
| walk-forward rolling validation | **don't** | the fixed three-way split is enough; goes to BACKLOG |

> **Conclusion**: your plan is missing only two things — **signal dedup** and
> **hypothesis alignment**. Together about 1 hour, but they correspond to the core
> mechanisms of RD-Agent and AlphaAgent respectively, and are hard currency in an
> interview. I'd add both to Phase 4.

---

# Appendix: a close read of the RD-Agent(Q) methodology

> Added in plan v3. This appendix records only the parts **directly portable to
> your project**, skipping the model branch and Co-STEER code generation (you use
> the spec approach, so you don't need them).
> Paper: `arxiv.org/abs/2505.15155` (NeurIPS 2025) · Code: `github.com/microsoft/RD-Agent`

## A. Two stages + feedback

<cite index="26-1">The Research stage dynamically generates goal-aligned prompts, forms hypotheses from domain priors, and maps them to concrete tasks; the Development stage has a code-generation agent implement the code and backtest it on a real market; the two stages are connected by a feedback stage that evaluates results and guides the next iteration, using a multi-armed-bandit scheduler for adaptive direction selection.</cite>

**Takeaway**: feedback is **a separate stage**, not an appendage of the backtest.
Your `analyst.py` corresponds to it, so it should have its own module and its own
tests.

## B. Synthesis's generation function G(H, F)

The paper writes hypothesis generation as `G(H_t^(a), F_t^(a))`:
- `H` = the **historical hypotheses** under that action type
- `F` = the corresponding **feedback**
- `a` = the **action type** (in RD-Agent, "change factor / change model")

**Ported to your project**: `a` degenerates to "which theme to attack" (one of the
five themes). So `knowledge/` and `trials.jsonl` must be queryable **sliced by
theme** — the agent asks "what have I tried in the risk theme, and where did it
fail", not "all history". This detail significantly affects prompt quality.

## C. SOTA set and the knowledge forest

<cite index="27-1">Each round scores the hypothesis, task, and experiment results on multiple dimensions; if the experiment beats SOTA under the current action type, its result is added to the corresponding SOTA set. The unit then diagnoses failed strategies and generates targeted improvement suggestions, fed back to Synthesis to guide the next hypotheses.</cite>

<cite index="22-1">The system maintains a "knowledge forest" holding the hypotheses, implementations, and results of all past experiments, adaptively adjusting hypothesis complexity by recent performance.</cite>

**Two portable points**:
1. **Maintain a separate SOTA per action type.** For you that is **one SOTA per
   theme**, not one global SOTA. So the agent knows "the current best in the
   inflation theme".
2. **Adaptively adjust complexity by recent performance**: realize this as one
   prompt rule — if the last N rounds were all rejected for overfitting → the next
   round forcibly lowers complexity (fewer parameters / simpler template).

## D. Validation's dedup methodology

<cite index="25-1">Factors are deduped first, computing their correlation against the existing SOTA factor library to filter redundant signals; surviving candidates are paired with the current best model and backtested in a production-grade market simulation.</cite>

**Note the order**: dedup is **before** the backtest. The backtest is the slowest
step.

**Threshold**: the original uses IC similarity with a threshold near 1.0 (drop only
near-identical). You are single-asset with a small factor base, so **0.8** is
recommended — a more aggressive dedup.

## E. Budget and timeouts (copy the magnitude directly)

<cite index="27-1">Co-STEER's inner optimization loop is capped at 10 iterations per task for both factor and model workflows; the Implementation Unit is limited to 600 seconds per task, the Validation Unit to 3600 seconds.</cite>

**Ported to your project**:
- inner loop (parameter-validation feedback) capped at **3** (you have no code
  generation, so you don't need 10)
- a single spec's grid scan should finish in **30 seconds** (9–27 cells × pandas
  vectorization); a timeout means a bug in the implementation, not big data
- global `MAX_TRIALS` 30–50

## F. Leakage protection is architecture, not discipline

<cite index="22-1">The data-centric design prevents the LLM from accessing raw market data or explicit time splits, mitigating information leakage; but this is also a design constraint that limits certain forms of reasoning.</cite>

**This line is worth copying into your PRD**: leakage protection relies on "the LLM
simply cannot get it", not "we agree not to give it". Your C2 (`split` accepts only
train/val, test needs a separate script + explicit flag) is the same idea.

Note the paper itself admits this **has a cost** — the agent therefore cannot reason
about temporal structure. You can honestly note the same in your Limitations.

## G. The paper's three self-admitted limitations (benchmarks for your Limitations)

<cite index="22-1">First, the system currently uses only the LLM's internal financial knowledge to form hypotheses, with no live news, alternative data, or structured domain priors; second, the data-centric design mitigates leakage but limits certain reasoning; third, the backtest assumes a long-short strategy on CSI 300 and does not fully account for market impact and capacity.</cite>

**What this suggests for you**: on the first point you actually **do better than
it** — you have `knowledge/rules/` and the five-theme framework, genuine structured
domain priors (the paper's future work). This is worth pointing out explicitly in
the PRD; it is a real difference from the paper, not hype.
