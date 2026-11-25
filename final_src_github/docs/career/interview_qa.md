# Interview Q&A

Twenty questions and answers, calibrated for a **Summer 2026 Data Science internship** with a quantitative-research internship as the anchor experience. Every answer is written to be spoken, not read — practice each one until you can deliver the *shape* of the answer without memorizing the exact words.

The three narrative anchors that recur across answers:
- **Hypothesis-first, then data validates or kills it** — this is the DS mindset.
- **Interpretability over black-box accuracy** — the reason for no ML/DL.
- **Honest metrics** — chose to report 0.94 Sharpe instead of an inflated 2.01.

---

## Behavioral / narrative

### 1. Tell me about yourself.

> I'm a first-year Master's student in Computer Science at the University of Chicago, graduating March 2028. My background is a mix of computer science and applied quantitative work. Before UChicago I spent six months as a full-time quantitative research intern at a Chinese systematic asset manager, where I built an end-to-end factor-research pipeline for gold trading — I did every stage from vendor data ingestion through cost-aware backtesting. What I took away from that project is that I really enjoy the hypothesis-driven, "form a question and let the data answer it" shape of work, which is why I'm targeting data-science internships for Summer 2026, ideally in Chicago or the Bay Area.

**Delivery notes:** 25–30 seconds. End with what you want next so the interviewer knows the ask. Do not launch into project details unprompted — the follow-up "tell me more" is the invitation.

---

### 2. Walk me through your gold factor project end-to-end.

> Sure. The problem was: given 50-plus macro and market indicators, can we build a transparent, hypothesis-driven signal for the direction of physical gold, and does that signal survive realistic trading costs? I owned every stage.
>
> Data ingestion — I refactored a fragile vendor API crawler into an idempotent, resumable script and moved all session credentials to environment variables. I also pulled some indicators from FRED and Yahoo Finance.
>
> Alignment — the raw data came in three frequencies: daily for yields and prices, weekly for some sentiment indices, and monthly for CPI and ETF-flow totals. I wrote the layer that shifts each series forward by its publication delay, forward-fills it onto the trading calendar, and audits how many rows were added versus dropped so silent data loss surfaces immediately.
>
> Exploratory QC — histograms, Q-Q plots, and hierarchically-clustered correlation heatmaps for every indicator, so I could catch bimodal distributions or redundant indicators *before* I wrote a single factor line.
>
> Factor library — I built eight rolling time-series operators — moving average, standard deviation, rank, correlation, linear-regression slope, and so on — plus five composition templates like "direction times intensity" or "signal times persistence". Every one of the 15 factors reduces to one of those templates applied to those operators, so the whole library fits on one page.
>
> IC diagnostics — I computed rolling 180-day Pearson and Spearman information coefficients for each factor across ten forward-return horizons, with an ICIR-based significance overlay on the heatmap.
>
> Backtest — three position-mapping styles run side-by-side, T+1 execution, 3-basis-point linear cost, compounded NAV, and a portfolio construction step that weights each factor by its realized single-name Sharpe.
>
> The final portfolio landed at Sharpe 0.94 against a volatility-normalized CSI 300 benchmark at 0.39, so 2.4x on Sharpe and 3.8x cumulative return at the same volatility budget.

**Delivery notes:** ~90 seconds. Six named stages, each one verb: refactored, wrote, ran, built, computed, ran. Do not go into every design decision — they will ask.

---

### 3. Why data science instead of continuing in quant?

> Two reasons, one temperamental and one technical. Temperamentally, I discovered during the internship that I don't love the emotional relationship with markets — the daily P&L swings, the "is my strategy still working today" anxiety. What I *do* love is the workflow: form a hypothesis about how the world works, express it as a testable formula, and let clean data validate or kill it. That workflow is the same in quant and in data science, but in data science the questions are more varied — pricing, product experimentation, user growth, forecasting — and the feedback loop is less noisy than daily market P&L.
>
> Technically, everything I built on this project — the ingestion pipeline, the data-quality QC, the rigorous rolling diagnostics, the anti-overfitting discipline — those are core data-science skills. This project is my proof I can do them under real constraints; I'd rather apply them to problems where the interpretability matters as much as the accuracy does.

**Delivery notes:** Do not badmouth quant. This answer works because it names what you *love*, not what you dislike.

---

### 4. Why did you not use any machine learning or deep learning?

> Two reasons. First, my mentor and I explicitly agreed that a signal we cannot explain is a signal we cannot govern. If a portfolio manager asks "why is the model long today?", we have to be able to answer "because the dollar theme's `pmf_069545` flipped negative on a CPI z-score shock, and it carries 8% of the risk budget." A gradient booster or an LSTM with the same in-sample Sharpe would have failed that operational bar. Second, on this dataset — around 8,000 daily observations, 50 indicators, one target — a well-designed handful of interpretable factors is competitive with anything ML would deliver, and it does not carry the overfitting risk. That is not universally true; if I had 500 features and 10 million rows, my answer would be different. This was a case where the constraints of the problem lined up with the value of interpretability.

**Delivery notes:** This is the question you *will* be asked in a DS interview. Prepare for the follow-up: "But when *would* you use ML?" Answer: "When the feature space is high-dimensional enough that hand-designed features leave signal on the table, and when the downstream user only cares about accuracy — not about being able to explain each prediction."

---

### 5. Tell me about a bug you caught that others missed.

> On the backtest engine, the first version of the portfolio construction applied a 5-day rolling-mean smoother to each single-factor position, and *then* smoothed the weighted sum again. Two rounds of smoothing over-blurred the signal, and portfolio Sharpe was coming out around 0.35 when the single-factor Sharpes were in the 0.4 to 0.7 range — the portfolio was somehow worse than its ingredients. That should be impossible for a positive-weight combination of positively correlated signals.
>
> I dug in, printed the intermediate position series, and realized the smoother was running on both the inputs and the output. The fix was to cache the per-factor positions *before* post-processing, do the weighted sum on those raw signals, and then run post-processing exactly once. Sharpe jumped from 0.35 to 0.94 — a 30x improvement from a two-line fix.
>
> The larger lesson: any pipeline that composes stages needs to be able to answer "does this stage assume its input is raw or already processed?" for every stage. Naming the intermediate objects clearly — I called them `directional_positions` versus `final_positions` — makes the answer physical.

**Delivery notes:** This is a golden answer because it demonstrates debugging instinct, quantifies the impact, and ends on a generalizable lesson. Practice this until you can deliver it without notes.

---

## Deep technical — project

### 6. What is Information Coefficient? Why Spearman instead of Pearson?

> IC is the rank correlation between today's factor value and the forward n-day return of the asset — it measures whether the factor systematically points in the right direction. Its running mean tells you the factor's edge; its standard deviation gives you the noise around that edge, and ICIR (IC divided by its standard deviation) is the t-statistic.
>
> I used Spearman as the primary metric for two reasons. First, robustness to outliers: gold has real fat-tailed days, and one 6-sigma move can dominate a Pearson correlation over a rolling window. Spearman only cares about rank order, so a single extreme observation gets bounded influence. Second, it captures monotonic relationships that are not necessarily linear, which is the right assumption for factors like the Aroon balance or percentile-mapped signals. I also computed Pearson IC for every factor and reported them side-by-side on the heatmap so a reader can see when the two disagree — a big disagreement is usually a sign that one or two outlier days are driving the Pearson value.

---

### 7. Why a 180-day rolling IC window? Why not full-sample, or 60 days?

> Three answers stack. Full-sample IC hides regime changes — the DXY–gold correlation was clearly regime-dependent between the 2019 rate-cut cycle and the 2022 hiking cycle, and a full-sample number averages those two regimes into an uninformative middle. Sixty days gives high time-resolution but the standard error on a 60-day rank correlation is too wide to reject the null for most factors. One hundred and eighty days lands in the middle: it is long enough that the standard error tightens, but short enough that a genuine regime break shows up in the heatmap within a quarter.
>
> There is also a business-context justification: 180 days is roughly the horizon over which a discretionary macro trader would tolerate a factor being under water without pulling the plug on it. Aligning the diagnostic window with the deployment expectation keeps the metric operationally honest.

---

### 8. How did you handle mixed-frequency data?

> The 50 indicators arrived at three frequencies: daily for prices and yields, weekly for some sentiment indices, monthly for CPI and ETF flows. Two subtle problems.
>
> First, look-ahead bias. A monthly CPI print for June isn't knowable until early July when it is released, so any factor computed on the June-labeled CPI value on a June trading day would be cheating. I built a `DELAY` parameter — 0 for daily, 3 for weekly, 7 for monthly — that shifts every reference date forward by the typical publication lag before any downstream code sees it.
>
> Second, calendar mismatch. Vendor CSVs use calendar days; my trading calendar has weekends and Chinese public holidays removed. After the delay shift, I forward-fill onto a contiguous daily grid and then drop non-trading days.
>
> Every file emits a processing log — how many rows added, how many dropped, how many NaN remained — so if a vendor changes their file format silently and one indicator suddenly returns zero rows, I know about it before it corrupts the factor table. I actually caught one such bug this way: the date-column detector was matching a spurious `time` column before the real `date` column, and one indicator was silently emitting empty CSVs. Would not have found it without the audit log.

---

### 9. Why did you try three position-mapping styles instead of picking one?

> To make the choice defensible instead of hand-tuned. Rolling percentile, tanh z-score, and clipped z-score are three reasonable ways to map an unbounded factor value into a bounded target position; each has different failure modes. Percentile is robust to outliers because it uses only rank information but throws away magnitude. Tanh z-score preserves magnitude smoothly but is sensitive to the tail of the distribution. Clipped z-score is a harder-edged variant that saturates faster.
>
> I ran all three side-by-side on every factor and every portfolio. Percentile came out on top in almost every case, which is consistent with the outlier robustness argument. But reporting all three means the interviewer — or a code reviewer — can see that I did not just pick the winner and hide the losers. That is the same discipline as A/B testing multiple variants in a product data-science context.

---

### 10. How did you construct the portfolio? Why not IC-weighted or equal-weighted?

> I tested all three schemes. Equal-weighted came in at Sharpe 0.40 — six of the fifteen factors had absolute IC below 0.01, meaning their trade direction was essentially a coin flip, so equal-weighting them just diluted the signal from the good factors. IC-weighted came in at 0.26, which was a surprise until I dug in: the highest-|IC| factor was a CPI-driven factor whose position mapping was all NaN because forward-filled monthly CPI has zero rolling standard deviation over 5-day windows. So IC-weighting was allocating the largest weight to a factor that could not produce a position — dead capital dressed up as diversification.
>
> Sharpe-weighted, with a soft floor at zero, landed at 0.94. It uses the only metric that has actually been validated against realized returns and costs. The floor drops any factor with negative in-sample Sharpe; I tested raising the floor to 0.3 to keep only high-conviction factors and it actually made things worse, because two factors with single-name Sharpe around 0.14 contributed enough diversification to outweigh the small noise they added. That was a nice counter-intuitive finding.

---

### 11. What is TWAP execution? How did smoothing help or hurt?

> TWAP — time-weighted average price — is the execution style where instead of rebalancing your whole target position in a single trade, you break it into equal-sized child orders spread over N time buckets. Its purpose in live trading is to minimize market impact.
>
> In my backtest the post-processing pipeline first sparsifies rebalances to every 10 trading days, producing a step function, and then applies a 5-day rolling mean. Analytically that is equivalent to a TWAP overlay: each 10-day rebalance is linearly spread over the following 5 days.
>
> Now, in the linear-cost model I used, this does not change total turnover — it spreads it out — so it does not directly help Sharpe. What it *does* do is make the code physically resemble how the strategy would trade in production, so the design is future-proof. Once you swap in a non-linear market-impact cost model, TWAP starts paying for itself because sliding 0.4 units of position over 5 days is much cheaper than dumping 0.4 units in one print.
>
> I tested reversing the two operations — smooth first, then sparsify — and Sharpe dropped by about 0.2. That was because smoothing before sparsification blurs the *signal* instead of the *execution*, throwing away information the factor was trying to send.

---

### 12. Why did you switch to compounded NAV? What went wrong before?

> The first version of the engine computed net asset value as `1 + net_return.cumsum()` — additive accumulation. In a normal market that produces the right answer to first order, but the issue is that under a drawdown of more than 100% the NAV goes negative, and once you try to compute a max drawdown from a NAV that has gone negative, you get nonsensical numbers like a 162% max drawdown. That is what I saw and what tipped me off.
>
> The fix is `(1 + net_return).cumprod()` — the standard compounded form. Once you do that, NAV is bounded below by zero and drawdowns are always in a sensible range. It also matters for correctly comparing to a compounded benchmark, and for annualization: the ratio of my strategy's arithmetic annualized return to its geometric compounded return is small on this data, but on more volatile strategies it can be meaningful.
>
> Beyond the bug itself, the lesson is that when a metric returns something impossible — a 162% drawdown, a negative Sharpe on a positive-return series, a probability outside [0, 1] — the right response is to trace back to the arithmetic and figure out which step violated an invariant. The number is the smoke alarm, not the fire.

---

## Statistics / methodology

### 13. How do you avoid look-ahead bias in a rolling backtest?

> Three places it can creep in. First, position mapping: the rolling percentile function excludes the current point from its own history — it looks at `window[:-1]`, not `window[:]`, so today's value cannot know its own percentile among itself. Second, execution: every position uses `position.shift(1)` before it is multiplied by the return, so yesterday's decision earns today's price move. Third, publication delays for lower-frequency indicators: the `DELAY` parameter in the alignment layer shifts every reference date forward by the release lag, so a monthly CPI print for June is never available on a June trading day.
>
> The one look-ahead I deliberately accept and disclose is the use of full-sample Spearman IC to fix trade direction. Rolling-IC direction was implemented and tested — it drops Sharpe to about 0.30, which is the honest lower bound on live performance for this factor family. I report both numbers.

---

### 14. What is your view on in-sample versus out-of-sample performance?

> Out-of-sample is the only number that matters for a deployment decision; in-sample is a lower bound on how much overfitting your process might have introduced, so it is diagnostic, not evaluative. On this project the 0.94 Sharpe is in-sample with respect to the direction decision but out-of-sample with respect to the rolling position mapping and the 3-basis-point cost model. If I were pitching this strategy to a real portfolio manager, I would run a walk-forward IC direction and quote whichever number that produces — I have implemented that variant, it produces around 0.30, and that number is what I would defend.
>
> The broader point is that overfitting is not a boolean — every quantitative process leaks *some* future information; the goal is to know exactly how much and to bracket the truth from below (walk-forward) and above (full-sample). I would rather report 0.30 to 0.94 as a range with named assumptions than either endpoint alone as a point estimate.

---

### 15. Why vol-normalize the CSI 300 benchmark instead of just comparing Sharpe?

> The two numbers I care about — cumulative return and maximum drawdown — are units-free but *not* volatility-free. If my strategy runs at 5.5% annualized volatility and I compare it to CSI 300 at 25.8% annualized volatility, I am silently comparing two things running at 5x different leverage. The CSI 300 will always look like it produces more cumulative return, because it is taking 5x more risk. That is not a fair fight.
>
> The fix is to scale the benchmark's *daily returns* to my strategy's volatility target — 5.5% — and *then* compound. It is critical to scale at the return level: if you take an already-compounded NAV curve and multiply its excess by a scalar, you preserve shape but not annualized volatility. That is a subtle bug and I actually made it before catching it.
>
> After vol-normalization, the CSI 300 benchmark comes in at 0.39 Sharpe versus my strategy at 0.94, and at the same 5.5% vol budget my strategy compounds to roughly 3.8x the benchmark's cumulative return with about 60% less maximum drawdown. That comparison is now honest.

---

## Design + reflection

### 16. What is the biggest mistake you made on this project?

> The double-smoothing bug I described earlier — I let it live for two weeks before catching it, and I was reporting a portfolio Sharpe of 0.35 when the real number should have been 0.94. The reason I did not catch it sooner was that I never asked the question "how is it possible that my portfolio Sharpe is lower than my best single-factor Sharpe?" That should have been an instant red flag, and I skated past it because the number was "close enough to plausible".
>
> The takeaway I have internalized is: when a composite metric is worse than its components, do not accept it. Positive-weighted combinations of positively-correlated signals should never underperform their best ingredient by more than a diversification adjustment. That is a hard constraint I now check explicitly, not implicitly.

---

### 17. If you had another month on this project, what would you do?

> Three things, in order of value. First, move the indicator selection into the pipeline itself. Right now the 50 indicators were curated by hand into 13 for the top-15 factors; if I auto-selected from the correlation-cluster output, the process would be reproducible and less mentor-anchored. Second, implement walk-forward IC for the direction decision, even at the cost of a lower reported Sharpe. Removing the last in-sample assumption is worth the honesty. Third, ship a non-linear cost model — impact roughly quadratic in trade size — so the TWAP smoother actually earns its keep. All three of those are Week-One work; the reason they did not ship was scope, not skill.

---

### 18. Explain the five-theme framework — why not just rank factors by Sharpe?

> Two reasons. One operational, one epistemic. Operationally, thematic grouping makes attribution explainable to a non-quant reader. If the portfolio manager asks "what is driving today's P&L?", "the ETF theme contributed 60 basis points, the Rates theme lost 20" is a sentence a human can act on. "Factors 7, 11, and 13 contributed 60 basis points" is not.
>
> Epistemically, ranking by Sharpe is a leaderboard — the top three factors are almost always the ones that got lucky in-sample on the specific data window I tested. Thematic construction forces me to keep at least one factor from each economic bucket even if its in-sample Sharpe is low, which improves out-of-sample robustness because the themes have genuinely different economic exposures. On this project the two lowest-Sharpe factors that made it into the final portfolio — around 0.13 Sharpe each — actually improved portfolio Sharpe because they added diversification worth more than the noise they contributed. Sharpe-ranking would have dropped them.
>
> More broadly, a data-scientist recruiter cares about the same thing: can you explain *why* your model made the decision it made? Thematic grouping is my scaffolding for that.

---

## Situational / behavioral

### 19. Tell me about a time you disagreed with your mentor.

> Early in the project my mentor pushed to use IC-weighted portfolio construction — weight each factor by the magnitude of its information coefficient. My gut said Sharpe-weighted would work better because IC and realized Sharpe are only loosely correlated on our factors. I did not push back verbally at that point; instead I ran both configurations in parallel and produced side-by-side portfolio metrics. IC-weighted came in at 0.26 Sharpe; Sharpe-weighted came in at 0.94. I brought both to the next review and let the numbers speak.
>
> He looked at it, agreed, and we moved on. In retrospect I could have raised the concern earlier and saved us a day of compute — the lesson I took was to voice a hypothesis before doing the work, not after, even when the person on the other side has more experience. In a DS setting that is asking "have we considered X?" in the design review, not saving it for the results readout.

---

### 20. How do you communicate complex quantitative findings to non-technical stakeholders?

> Three habits. One: lead with the number and the sentence that frames it — "at the same 5.5% volatility budget, the strategy compounds to 3.8x the benchmark" — before any methodology. Two: visualize the *right* comparison. My NAV chart shows the strategy curve *and* the vol-normalized benchmark on the same axis, because that is the comparison a reader will make automatically anyway, and if you do not draw the right benchmark, they will imagine a wrong one. Three: put the honest limitations in a labeled section, not buried in a footnote — the section I called "Honest limitations" in the methodology doc names the in-sample direction and the six-of-fifteen factor coverage gap, so a stakeholder has to actively skip past them to miss them.
>
> The meta-principle is: I want the reader to reach the correct conclusion without having to trust me. If the chart, the number, and the caveats are all on the page, they can judge the result independently. That is the same discipline as writing a good product data-science report — the reader should be able to evaluate your work without you in the room.
