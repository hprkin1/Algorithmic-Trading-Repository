# Correlation-triggered pairs reversion — results (2026-10-30)

Protocol, rules and reasoning: see the strategy doc (Claude Docs, "Correlation Pairs Strategy — Thought Process & Breakdown").
Raw numbers: `results.json`; basket growth by version: `basket_growth.csv`. Reproduce: `python scripts/research_pairs.py --holdout`.

## Verdict: no sleeve added. The original idea is retired.

Headline basket = EWA/EWC, XOM/CVX, KO/PEP, V/MA, HD/LOW, GLD/SLV, equal capital. Sharpe after 5 bp fee + 5 bp slippage per side and 0.5%/yr borrow:

| Version | DEV 2008–15 | VAL 2016–21 | HOLDOUT 2022– | Rule 1 (≥0.5 DEV & VAL) |
|---|---|---|---|---|
| A textbook hedged (gap ≥ 2σ) | 0.40 | −0.13 | −0.12 | fail |
| B gap + Henry's correlation trigger | 0.00 | −0.47 | 0.61 | fail; also fails rule 2 vs A |
| C Henry's original style (long laggard only) | 0.58 | 0.94 | 0.88 | passes **on paper** |

**Why C is rejected anyway:** on the exact same days C is invested, buying the *leader* instead gives Sharpe 0.58 / 0.86, buying *both* gives 0.64 / 1.02, and holding both all the time gives 0.62 / 0.97 (with 4–5× the return). The catch-up itself (laggard minus leader) earns 0.16% / 0.44% a year (Sharpe 0.11 / 0.39). C's Sharpe is long-equity exposure, not the idea.

- The correlation trigger cuts time in market from ~42% to ~12% and doesn't improve the hedged version in DEV or VAL. B's good holdout (0.61) is not supported by the earlier periods.
- Under A only EWA/EWC (0.18 / 0.07) and HD/LOW (0.12 / 0.71) were positive in both DEV and VAL, both weakly; HD/LOW then lost in the holdout (−0.24).
- Stability (A, DEV): Sharpe 0.01–0.48 across lookback 40/60/90 × entry 1.5/2/2.5; fragile.
- Controls as expected: QQQ/SPY hedged is negative in every period (A: −0.42 / −0.20 / −0.55): the gap trends. SPY/EWU A: 0.10 / −0.31 / −1.02.
