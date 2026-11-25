# Elevator pitch

Three scripts calibrated to three real settings. Rehearse each until you can deliver it without looking down — the *cadence* is what convinces, not the content.

---

## 30-second version (career-fair booth, recruiter is standing)

> Hi, I'm Sitong — I'm a Master's student in Computer Science at the University of Chicago, graduating March 2028, and I'm looking for a Summer 2026 data-science internship. I just finished a six-month quantitative research internship in Shanghai where I built an end-to-end data pipeline that ingests around 50 macro and market indicators, aligns them across three different sampling frequencies, and turns them into a signal for gold-price direction. My final model beats a volatility-matched benchmark by about a 2.4-times Sharpe ratio, and every design decision is reproducible from a single GitHub repo. I loved the hypothesis-driven side of that work; I'm looking for roles where I can bring that same rigor to product data science.

**Beats to hit:** name + school + timing → what I built (concrete) → result (one number) → what I want next.

---

## 45-second version (recruiter has follow-up capacity, or LinkedIn message intro)

> Hi, I'm Sitong Zhang — CS Master's at the University of Chicago, graduating March 2028, targeting Summer 2026 data-science internships, ideally in Chicago or the Bay Area. My last role was a full-time quantitative research internship at a systematic asset manager in Shanghai. I was the sole contributor on a factor-research pipeline for physical gold: I ingested 50-plus macro and market indicators from three different vendors, wrote the alignment layer that harmonizes daily, weekly, and monthly data to one trading calendar, designed 15 interpretable factors grouped into five economic themes, and validated them with rolling information coefficients and a cost-aware backtest.
>
> The final Sharpe-weighted portfolio delivered a Sharpe of 0.94 against a volatility-matched CSI 300 benchmark at 0.39 — so 2.4x on Sharpe, 3.8x on cumulative return at the same risk budget. What I really loved was the workflow itself: form a hypothesis about *why* the dollar or geopolitical stress should move gold, express it as a formula, and then let the data validate or kill it. That is the exact mindset I want to bring into a data-science role — building things that decision-makers can trust because the logic is legible.

**Beats to hit:** identity → project scope (three verbs: ingested, wrote, designed, validated) → numbers → *why this matters for a DS role*.

---

## 60-second version (informational chat, on-campus deep-dive, or dinner)

> I'm Sitong Zhang — I'm doing a Master's in Computer Science at the University of Chicago, graduating March 2028. I'm looking for Summer 2026 data-science internships, and I'm most interested in Chicago and Bay Area teams working on high-signal problems where interpretability matters.
>
> Between December 2025 and June 2026 I did a full-time quant research internship at a Chinese systematic asset manager. It was a solo project: build a research pipeline that takes 50-plus macro and market indicators — dollar index, real yields, ETF flows, geopolitical risk, US CPI, all of it — and turns them into a daily trading signal for physical gold. I did every stage. I refactored the vendor API crawler and moved credentials to environment variables. I wrote the alignment layer that shifts weekly and monthly indicators forward by their publication delays and forward-fills them onto the trading calendar. I designed 15 interpretable factors grouped into five economic themes, deliberately staying away from machine learning because the mentor and I agreed a signal you cannot explain is a signal you cannot govern. I built the IC diagnostic layer for rolling 180-day rank correlations, and the backtest engine with T+1 execution and a linear cost model.
>
> The final Sharpe-weighted portfolio came in at 0.94 Sharpe versus a volatility-normalized CSI 300 benchmark at 0.39 — so 2.4x Sharpe, 3.8x cumulative return at the same volatility budget. I could have reported a much higher number from an earlier version, but I caught two bugs — a double-smoothing bug and an incorrect NAV compounding — that had been inflating it, and I would rather ship a 0.94 I can defend than a 2.01 I cannot.
>
> What I took away is that I really enjoy this shape of work: the hypothesis first, then the data test, then the honest report. That is exactly what data science is at its best, and it is why I am moving toward DS roles rather than pure trading. The stress of markets does not motivate me the way finding a defensible answer to a well-framed question does.

**Beats to hit:** identity → what I did (concrete, one-verb-per-clause) → one bug that shows judgment → tie back to DS narrative.

---

## Delivery notes

- **The two numbers you must be able to say without hesitation:** *"Sharpe 0.94, 2.4x the vol-normalized benchmark."* Everything else is optional; those two land the story.
- **Do not say the internal firm's name unless directly asked.** *"A Chinese systematic asset manager"* is short, credible, and unambiguous.
- **When you feel the person's attention going, cut.** The 45-second version is the safest default. The 60-second version is only for people who explicitly opened the door.
- **The pivot line.** *"The stress of markets does not motivate me the way finding a defensible answer to a well-framed question does."* This is your bridge from a quant background to a DS ask. Rehearse it enough to say naturally, not defensively.
- **After you deliver a pitch, hand them a question.** *"Are you hiring for Chicago or Bay Area DS teams this summer?"* or *"What does your team look for in a summer intern?"* — turn the monologue into a conversation.
