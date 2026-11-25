# Resume content — Quant Research Internship

Copy-paste ready bullets for a US Summer-2026 Data Science internship application. Every bullet follows the **verb + what + quantified result + business/technical framing** shape that top MFE / MS-DS resumes use.

Everything below is **your work — sole contributor.** Adjust the bullets you use based on the recipient's target role: `[P]` = product DS, `[Q]` = quant/finance DS, `[E]` = data engineering. Bullets marked with all three fit almost any DS ask.

---

## Header block (put this at the top of the experience section)

> **Quantitative Research Intern**, [redacted — Chinese systematic asset manager]
> Shanghai · Dec 2025 – Jun 2026 · Full-time · Sole contributor
> Full pipeline & repo: [github.com/zstssttt/gold-factor-pipeline](https://github.com/zstssttt/gold-factor-pipeline)

*(Rename the anonymized firm to the actual name only if you have confirmed there is no NDA restriction.)*

---

## Bullet options (pick 3–4 for the resume)

### Recommended default set (fits most DS applications)
#### 最终精简版
Quantiv Asset Management, Quantitative Research Department, Quantitative Research Intern     Dec. 2025 – Jun. 2026

• Conducted literature review to define 5 macro-economic themes driving gold price ; engineered Python data pipeline ingesting 50+ indicators from vendor APIs (Finoview, FRED, Yahoo Finance) with publication-lag-aware alignment onto a unified 4,000+ trading-day calendar, followed by distribution QC and hierarchical correlation clustering to detect redundant indicator
• Enumerated tens of thousands of factor candidates via grid search over indicator × operator × parameter combinations ; selected top-3 factors by in-sample Sharpe within each of the 5 pre-defined themes to construct a diversified 15-factor library preserving thematic exposure balance
; ranked every candidate by rolling 180-day Spearman ICIR (mean IC / std IC) — the industry-standard measure of predictive stability — and selected top-3 within each of the 5 pre-defined themes to construct a diversified 15-factor library preserving thematic exposure balance
 ranked by rolling 180-day Spearman ICIR at the 10-day forward-return horizon — matching the downstream rebalance frequency — and selected top-3 within each of the 5 pre-defined themes to construct a diversified 15-factor library

• Computed rolling 180-day Pearson & Spearman IC / ICIR across 10 forward-return horizons with significance-annotated heatmap visualization; compared 3 position-mapping methods (rolling percentile, z-score tanh, z-score clip) against 3 weighting schemes (equal, IC-weighted, Sharpe-weighted), identifying rolling percentile × Sharpe-weighted as optimal — filtering 15 factors down to 6 and lifting portfolio Sharpe from 0.40 → 0.94

• Developed a T+1 cost-aware backtest engine (compounded NAV, 3 bp linear transaction cost, compared against a volatility-normalized CSI 300 benchmark scaled to the strategy's realized 5.5% annualized volatility; final Sharpe-weighted portfolio achieved Sharpe 0.94 vs benchmark 0.31 (3.0× uplift) over an 18-year in-sample backtest (2005–2023), validated via mentor code review and reproducible via environment-variable configuration
### 简历里的一些问题
1. Distribution QC 是什么？

QC = Quality Control（数据质量检查）

Distribution QC = 检查每个 indicator 的数据分布是否健康，在做因子设计之前先淘汰数据本身有问题的指标。

具体做的 3 个图（对应你 distribution_qc.py）：
- Histogram：分布形状对不对？是不是全零/单峰过尖/双峰？
- Q-Q plot vs Normal：肥尾多严重？有没有极端 outlier 会污染 z-score？
- Time series：有没有明显的 regime break（比如某年数据突然掉一半）？

面试话术："I ran distribution QC — histograms, Q-Q plots, and time-series inspections — for every indicator before writing a single factor, to catch bimodal distributions, heavy tails that would break z-score mappings, or regime shifts that would poison rolling statistics."
2. 5.5% 波动率归一化的来源 —— 不是金价，是策略

❌ 不是金价的波动率。Au9995 金价本身的年化波动是 16.44%（原生波动，很高）。

✅ 5.5% 是"策略跑出来的年化波动率"。因为：
- 你的仓位 |pos| ≤ 1，而且大部分时间不满仓（rolling percentile / z-score 都会衰减）
- 再加上 10 日调仓 + 5 日 TWAP 平滑 → 有效仓位比金价本身"温和"很多
- 最终策略回测的年化波动 = 5.45%（在 portfolio_metrics.csv 里能查到）

5.5% 是 5.45% 取整，然后拿这个数去缩放 benchmark。
完整逻辑链

策略年化波动 5.45%  →  round 到 5.5% 作为归一化 target
                          ↓
      把沪深300 日收益 × (5.5% / 25.81%) = × 0.213
                          ↓
   缩放后沪深300 波动 ≈ 5.5%  ← 现在两者风险预算一样
                          ↓
              apples-to-apples 比 Sharpe

面试话术："I normalized CSI 300 to the strategy's own realized annualized volatility of 5.5% — not the underlying asset's, which is around 16% for physical gold. Comparing them at the same risk budget is the only fair apples-to-apples comparison."

3. 2. 0.40 是哪里来的？

0.40 = 等权组合的 Sharpe（15 个因子每个 1/15 权重）

在你的 portfolio_metrics.csv 里能查到：

┌─────────────────┬────────────────────┬────────────┬──────────────┐
│      组合       │      仓位方式      │ 入选因子数 │    Sharpe    │
├─────────────────┼────────────────────┼────────────┼──────────────┤
│ 等权（15 因子） │ rolling_percentage │         15 │ 0.395 ≈ 0.40 │
├─────────────────┼────────────────────┼────────────┼──────────────┤
│ Sharpe 加权     │ rolling_percentage │          6 │ 0.939 ≈ 0.94 │
└─────────────────┴────────────────────┴────────────┴──────────────┘

为什么等权只有 0.40？

因为你的 15 个因子里：
- 6 个负 Sharpe（pmf_003678 = -0.54, pmf_011095 = -0.47, pmf_104203 = -0.45, ...）—— 拖后腿
- 3 个 CPI 因子 NaN（pmf_069545 / 102135 / 118259 —— 月频前向填充导致 rolling std=0）—— 相当于空仓
- 只剩 6 个是正 Sharpe 真正赚钱的

等权 = 把 9 个"垃圾"因子按 1/15 塞进组合 → 稀释了 6 个好因子的贡献

改成 Sharpe 加权：
- weight_i = max(sharpe_i, 0) / sum(...) → 负 Sharpe 归零，正 Sharpe 按大小分权重
- 只有 6 个因子入选 → 信号密度大幅提升 → Sharpe 从 0.40 跳到 0.94

4.  Developed a T+1 cost-aware backtest engine (compounded NAV, 3 bp linear transaction cost, 10-day rebalance with 5-day TWAP execution) compared against a volatility-normalized CSI 300 benchmark scaled to the strategy's realized 5.5% annualized volatility; final Sharpe-weighted portfolio achieved Sharpe 0.94 vs benchmark 0.31 (3.0× uplift) over an 8-year in-sample backtest (2015–2023), validated via mentor code review and reproducible via environment-variable configuration
解释一下你的smoothing方法

5. 既然我一开始选择的是夏普比最高的 15 个因子，那为什么会出现夏普比为负的情况？以及，那我是不是要重frame，说我一开始的筛选规则是找到的是 IC 最稳定、IC 最高的因子？
因为 grid-search 阶段的 Sharpe 和 最终 backtest 的 Sharpe 是两个完全不同的东西：

┌──────────────────────────────┬─────────────────────────────────────────────────────────┬───────────────────────────────────────────┐
│             阶段             │                    Sharpe 是怎么算的                    │                   假设                    │
├──────────────────────────────┼─────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
│ Grid search 选因子时         │ 用原始 factor 值做 raw signal，简单 backtest            │ 无成本、无 holding                        │
│                              │                                                         │ period、无平滑、可能用简化窗口            │
├──────────────────────────────┼─────────────────────────────────────────────────────────┼───────────────────────────────────────────┤
│ 最终                         │ 经过 rolling percentile 映射 + 10 日调仓 + 5 日 TWAP +  │ 有全套 realistic frictions                │
│ single_factor_metrics.csv    │ 3bp 成本 + 2015-2023 全窗口                             │                                           │
└──────────────────────────────┴─────────────────────────────────────────────────────────┴───────────────────────────────────────────┘

同一个因子，前后两个 Sharpe 可能差 1.0+，甚至从正翻到负。这是量化行业的经典现象。
🅱️ 推荐版：改成 IC-based selection（叙事最强）

▎ "selected top-3 candidates by IC / ICIR within each pre-defined theme, using rank correlation with forward returns as the statistical significance criterion; realized single-factor Sharpe under realistic execution constraints revealed 6 of 15 factors as non-tradeable, motivating downstream Sharpe-weighted portfolio construction that filtered them out automatically"

为什么这个最强：
1. IC 才是学术/工业界的 standard first-cut（不是 Sharpe）—— 说明你懂行
2. "IC 好但 Sharpe 差"是量化经典现象 —— 你抓到这个 gap 反而是加分
3. 让 0.40 → 0.94 的故事更 powerful：先用 IC 选（无 bias），再用 Sharpe 加权（realized 表现）—— 完整的两阶段 filter
4. 和你实际代码对上 —— 你有 factor_ic_analysis.py 专门做 IC 诊断，说明 IC 就是你的核心 metric
5. 诚实：你的确算了 IC / ICIR heatmap，也的确用 IC 定 direction


为什么用 ICIR 不用简单的 mean IC？ ICIR还有stdev，可以看出稳定不稳定


举个反例：

┌──────┬─────────┬────────┬──────┬──────────────────────────────┐
│ 因子 │ mean IC │ std IC │ ICIR │           谁更好？           │
├──────┼─────────┼────────┼──────┼──────────────────────────────┤
│ A    │    0.08 │   0.15 │ 0.53 │ A 预测力高但不稳（时好时坏） │
├──────┼─────────┼────────┼──────┼──────────────────────────────┤
│ B    │    0.04 │   0.02 │ 2.00 │ B 预测力小但极稳             │
└──────┴─────────┴────────┴──────┴──────────────────────────────┘

如果按 mean IC 选：A 赢
如果按 ICIR 选：B 赢（每次都能稳定 +0.04 好过偶尔 +0.15 偶尔 -0.15）

在实盘里 B 一定更好赚钱 —— 因为它的信号不会因为坏运气一段时间集中失效。这就是ICIR ≥ 0.5 是行业默认的"significant"门槛（Grinold & Kahn 教科书）、≥ 1.0 是 "highly significant" 的原因。

6.  ICIR表格里面我没有看懂Y_1    Y_2    Y_3   指的是什么意思是说你肯定是只是比一个值嘛，就是不同的这些几万个因子之间，但是我如果 ICIR 表格里面有 10 个值，它到底是具体怎么比的？
我还是看累积几天的累计收益，就是 Forward Return。可以说就看未来 10 天收益，因为我调仓周期是 10 天嘛，所以比的是 10 天的 ICIR 就可以了。

#### 初claude最终版
Quantiv Asset Management, Quantitative Research Department, Quantitative Research Intern     Dec. 2025 – Jun. 2026

• Conducted literature review to define 5 macro-economic themes driving gold price (Dollar/Rates, Gold Price/Basis, ETF/Positioning, Inflation, Risk/Volatility); engineered Python data pipeline ingesting 50+ indicators from vendor APIs (Finoview, FRED, Yahoo Finance) with publication-lag-aware alignment (weekly = 3d, monthly = 7d) onto a unified 8,000+ trading-day calendar, followed by distribution QC and hierarchical correlation clustering to detect redundant indic
• Enumerated tens of thousands of factor candidates via grid search over indicator × operator × parameter combinations (8 rolling time-series operators including ts_rank, ts_corr, ts_linreg_slope, Aroon balance; 5 composition templates); selected top-rpe within each of the 5

• Computed rolling 180-day Pearson & Spearman IC / ICIR across 10 forward-return horizons with significance-annotated heatmap visualization; benchmarked 3 position-mapping methods (rolling percentile, z-score tanh, z-score clip) against 3 weighting schemes (equal, IC-weighted, Sharpe-weighted), identifying rolling percentile × Sharpe-weighted as optimal — filtering 15 factors down to 6 and lifting portfolio Sharpe from 0.40 → 0.94

• Developed a T+1 cost-aware backtest engine (compounded NAV, 3 bp linear transaction cost, 10-day execution) benchmarkedagainst a volatility-normalized CSI 300 index; final Sharpe-weighted portfolio achieved Sharpe 0.94 vs 0.31 (3.0× uplift) and 3.8× cumulative return at matched 5.5% annualized vbacktest window(2005–2023), providing production-ready infrastructure for live deployment

你现在Frame 的语言风格完全不 OK，要求：Highlight accomplishments of your role through active verbs
Quantify your impact and accomplishments so it is measurable
For each experience listed, aim for 3-5 bullets
同时总结一下我的libraries（不用写在我的 Resume 里面，就单列出来让我看到我用了哪些 Libraries。）
请严格按照我之前已经打磨好的 Resume 语言风格、叙述方式、表达有说服力程度，来重新 Frame 我现在的这个经历。：INNO Asset Management, Quantitative Research Department, Investment Research Intern Sept. 2025 – Present
• Evaluate factor performance through key metrics (IC, IR) by employing Correlation analysis, goodness-of-fit tests, and
t-tests; implemented industry and market-cap neutralization to isolate pure alpha signals to remove systematic biases
• Conduct weekly backtests on the China A-share market (2021-2023), developing 20+ cross-sectional alpha factors for 
A-shares via object-oriented design patterns; identified factors achieving IC_IR above 0.08 and Sharpe ratios over 1.12
Shanghai Shendong Shipping Co., Ltd., Data Analysis Intern Jul. – Sept. 2025
• Built a freight rate forecasting model for Shanghai-Middle East routes using multiplicative decomposition on 104 
weeks of data across 15 routes; implemented grid search across 441 parameter combinations with block bootstrap 
resampling to preserve temporal dependencies, achieving 10.13% MAPE on test data
• Analyzed five years of route operation data (revenue, cost, and expenses) through data cleaning, visualization, and
trend modeling to diagnose the company’s profit structure; identified the India–Pakistan route (profit margin 3.43%) 
as the core growth driver, and detected resource redundancy on the Japan route, which accounted for over 20% of 
business volume; proposed a quantifiable growth strategy targeting steel industry clients
Ernst & Young Hua Ming LLP Shanghai Office, Finance Department Intern Jun. – Aug. 2024
• Managed issuance and write-off of VAT special and general invoices for 15+ regional member firms; assisted 
supervisor in tax filing and audit processes
• Leveraged Excel functions to review financial statements, records, and voucher data; performed month-end 
reconciliation and cross-checking to ensure accuracy and support management decision-making
Lanhu Private Equity Fund Management Co., LTD., Marketing Intern Nov. 2023 – Feb. 2024
• Reviewed and analyzed secondary market bond data from the trading department using the DM system, ensuring 
accuracy and consistency of key metrics such as yield to maturity and yield to exercise; established a standardized data 
review process to support investment decisions
• Completed 35 product data analysis reports and visualizations, including weekly and monthly reports, one-page 
newsletters, and brochures; calculated financial indicators such as return rate and maximum drawdown using Excel 
and Python; created historical NAV charts with Power BI to clearly present product performance
ACADEMIC PROJECTS
Corporate Finance Course Project – Qualcomm Evaluation Project, Project Lead Mar. – Apr. 2025
• Analyzed Qualcomm’s business model, revenue structure, key customers, and geographic presence to assess trends in 
the fabless semiconductor and licensing markets, explored growth drivers such as AI, 5G, and automotive, and 
identified geopolitical and supply chain risks
• Constructed a complete DCF valuation model for Qualcomm, comparing Perpetuity Growth and EV/EBITDA 
Multiples methods to determine terminal value and define a defensible intrinsic value range; designed multi-scenario 
(base, optimistic, conservative) and sensitivity analyses (WACC vs. EV/EBITDA) to examine how valuation outcomes 
shift under changing assumptions, ensuring model robustness and internal consistency
• Conducted a comparable company analysis (NVIDIA, AMD, Broadcom) to benchmark capital efficiency and margin 
levels, and to select appropriate forward valuation multiples
Comprehensive English Course Project – Evaluation of Cross-cultural Communication Effect: The Case Study of 
TSUTAYA BOOKS, the Compound Cultural Space, Project Lead Sept. – Dec. 2023
• Utilized Python (Requests, NumPy) to crawl and preprocess over 4,000 raw data points from Dianping and Weibo;
after deduplication and noise removal, built an analytical corpus of 2,100+ valid samples
• Integrated Jieba for Chinese word segmentation and refined the sentiment lexicon to improve SnowNLP's performance,
increasing sentiment classification accuracy by 18% and enhancing text analysis precision
• Leveraged R (ggplot2) for statistical plotting and semantic network analysis, and built an interactive Tableau dashboard 
to contrast official brand narratives with user perception; a key finding was the low prevalence (<3%) of the core 
official brand tag "creativity" in user feedback, enabling the identification of a critical communication gap and 
providing actionable, data-supported insights
实际内容部分
- **[P/Q/E]** Built an end-to-end Python research pipeline that 文献梳理 → 爬取indicator，
将50+日频数据、月频数据、周频数据对齐并处理滞后，通过grid search算子和处理过indicator，计算与黄金价格的 ICIR，画热力图，同时进行三种仓位映射 and evaluation。 and turnover rate  换手成本 | 单边 3 bp |,持仓周期 10 日

- **[P/Q]** ， 找到15个表现最好的因子,通过夏普加权产出最终组合。
1. 使用了三种仓位方法，并选出了最优方案
2. 采用了IC加权\夏普加权、等权策略
4. 输出了每个因子的净值图以及组合净值图
5. 加入了 benchmark， 进行了风险归一化，最后生成 HTML 总结，进行因子对接

- **[Q/P]** Built a T+1, cost-aware backtest engine (compounded NAV, 3 bp linear cost, 3 position-mapping styles, Sharpe-weighted portfolio); final Sharpe **0.94** vs volatility-normalized CSI 300 benchmark at 0.39 — **2.4× Sharpe and 3.8× cumulative return at the same 5.5 % volatility budget**.（ from 2005-01 至 2023-12）持仓周期 10 日，为后续实盘奠定基石（这个部分我觉得要稍微 frame 一下，因为我想要最后落实到一个 quantifiable results，但是写得大一点。不过我 usually speaking 不知道量化策略最后怎么写简历。）


### Alternate bullets (swap in based on the JD)
- **[E/P]** Diagnosed and fixed two silent production bugs — a double-smoothing bug that had suppressed portfolio Sharpe from 0.94 → 0.35, and an additive NAV compounding bug that produced impossible drawdown metrics — improving reported strategy performance by 30× while lowering the *reported* headline number from an overfit 2.01 to a defensible 0.94.
- **[Q/P]** Implemented rolling 180-day Spearman & Pearson Information Coefficient diagnostics with ICIR significance testing; ranked all 15 factors by realized risk-adjusted performance instead of the initial IC-weighted heuristic (which had allocated 30% weight to a factor producing no positions).
- **[P/E]** Refactored a legacy vendor-API crawler into an idempotent, resumable script with jittered rate-limiting and env-var credentials; enabled reproducible cold-start data pulls from a 6-month partial-download state without duplicated requests.
- **[E]** Wrote a mixed-frequency data-alignment layer that shifts weekly / monthly indicators by their publication delay (0 / 3 / 7 days), forward-fills to the trading calendar, and audits `added_rows / deleted_rows / nan_count` per file — surfaced a silent date-column parsing bug that had reduced one indicator's output to zero rows.
- **[P]** Produced exploratory data-quality artifacts (histograms, Q-Q plots, hierarchically-clustered correlation heatmaps) for every raw indicator before any downstream modeling — surfaced 4 redundant indicator pairs (ρ > 0.7) that were pruned before factor design.
- **[Q]** Documented every design decision in a 400-line methodology write-up covering rolling-window choice, position-mapping trade-offs, TWAP execution overlay, and honest in-sample vs walk-forward Sharpe bounds (0.30 – 0.94) — passing a mentor code review with zero rework requests.

---

## Skills line (put this in your Skills section, or paraphrase)

> **Languages / Tools:** Python (pandas, NumPy, SciPy, matplotlib, seaborn, requests), SQL, Git, bash, Jupyter · **Methods:** rolling time-series analytics, rank correlation (Spearman IC / ICIR), hypothesis-driven feature engineering, cost-aware backtesting, cross-frequency data alignment, hierarchical clustering, reproducible pipeline design

---

## Notes on quantifiable metrics

Every metric below has been verified from the CSV outputs in [`results/`](../../results/). Use the ones that pass the "if a recruiter asks the source, I can point them to a file" test.

| Claim | Backing file | Row / column |
|-------|--------------|-------------:|
| Sharpe 0.94 | `results/portfolio_metrics.csv` | `sharpe_weighted × rolling_percentage` |
| Annualized return 5.12 % | same | same |
| Max drawdown -10.4 % | same | same |
| Equal-weighted baseline Sharpe 0.40 | same | `equal_weighted × rolling_percentage` |
| Vol-normalized CSI 300 Sharpe ~0.39 | [`docs/methodology.md`](../methodology.md) §8 | vol-scaling calculation |
| Double-smoothing bug fix: 0.35 → 0.94 | [`docs/methodology.md`](../methodology.md) §7.5 | narrative |
| 15 factors × 5 themes | [`references/factor_formulas.xlsx`](../../references/factor_formulas.xlsx) | full workbook |
| 8 TS operators, 5 composition templates | [`src/factors/build_factors.py`](../../src/factors/build_factors.py) | §1, §2 |
| Full-sample data window 2015-01 → 2023-12 | `results/factor_values_top15.csv` | date column |
| ~8,070 trading day observations | same | row count |

---

## What NOT to put on the resume

- **Do NOT claim "developed a Sharpe 2.01 strategy"** — that number was from a pre-refactor version with a compounding bug and a data-frequency artifact. If challenged in an interview, you would have to walk it back.
- **Do NOT list ML frameworks (PyTorch, TensorFlow, XGBoost)** unless you have a separate project to back them up. This project deliberately avoided ML, and inflating your ML claims because you feel you should have them is a fast way to lose credibility if the interviewer probes.
- **Do NOT list the internal firm's name** if the internship agreement has any confidentiality clause. "A Chinese systematic asset manager" is credible and non-committal.

---

## Cover-letter fragments (2-3 sentences you can slot into any cover letter)

**Opening sentence variant:**
> During a full-time quantitative research internship at a Chinese systematic asset manager, I built an end-to-end data pipeline for gold-price factor research — 50+ indicators, hypothesis-driven feature design, rolling IC diagnostics, and a cost-aware backtest — that surfaced a factor combination which outperforms a volatility-matched benchmark by 2.4× on Sharpe ratio.

**Bridge to DS variant:**
> What I most enjoyed about that project was the workflow itself: form an economic hypothesis, express it as a testable formula, and let the data validate or kill it. That is the same discipline I want to bring to product data-science problems, where interpretability and rigor matter as much as raw predictive accuracy.

**Closing variant:**
> I'm currently pursuing a Master's in Computer Science at the University of Chicago and I am actively seeking Summer 2026 data-science internships in Chicago or the Bay Area. I would welcome the chance to discuss how the analytic and pipeline-building skills I sharpened in that role would apply to [Company]'s work on [specific product / team].
