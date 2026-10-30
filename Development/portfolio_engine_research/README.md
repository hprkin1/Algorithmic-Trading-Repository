# portfolio-engine research archive

Strategies researched with [portfolio-engine](https://github.com/hprkin1/portfolio-engine) that **failed** their
pre-declared tests or were **retired**. Kept here with their results so the reasoning isn't lost.
The live engine and live candidates (BTC OBV momentum, BTC AC momentum, oil trend/carry) stay in portfolio-engine.

| Folder | Idea | Outcome |
|---|---|---|
| `oil_correlation_original/` | Original `CorrArb_Class`: buy the laggard when WTI/Brent correlation < 0.6 | Retired: never trades on clean data; its one profit was the Apr-2020 −$37 print |
| `oil_spread_reversion/` | Brent–WTI z-score spread, market-neutral | Failed: futures profit was a roll artefact; loses on USO/BNO |
| `pairs_v1/` | 6 hand-picked equity pairs; textbook / correlation-trigger / long-laggard versions | Failed: hedged versions < 0.5 Sharpe; long-laggard return is just equity beta |
| `pairs_v2/` | 20 pairs re-chosen yearly from 213 stocks/ETFs, β hedge, ADF + half-life, correlation-break exit | Failed: Sharpe −0.59 after costs (0.29 before) |
| `multi_asset_trend/` | Time-series trend on 18 ETFs (1/3/12m, vol-scaled, monthly) | Failed: Sharpe 0.20 vs bar 0.5 |

Full thought process and charts for the pairs work: the "Correlation Pairs Strategy — Thought Process & Breakdown" doc.

## Running any of these

The code depends on portfolio-engine's `engine` package. Clone portfolio-engine, copy the strategy file(s) into its
`strategies/` folder (for pairs_v2 both `pairs_portfolio.py` and `pairs_universe.py`; pairs_v2 also needs
`pip install statsmodels`), copy the script into `scripts/`, then run from the portfolio-engine root, e.g.
`python scripts/research_pairs_v2.py`. Scripts that import a strategy from `strategies` by name need it added to
`strategies/__init__.py`.
