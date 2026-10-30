# Pairs v2 (systematic yearly selection): results 2026-10-30

Plan fixed before results: see the strategy doc. Reproduce: `python scripts/research_pairs_v2.py`.
Universe: 213 US stocks/ETFs in 46 groups (`strategies/pairs_universe.py`); pairs chosen on year N
(β hedge, ADF p<0.10, half-life ≤30d, top-20 by distance), traded in year N+1. Costs 5+5 bp per side, 0.5%/yr borrow.

## Verdict: fails the bar. Not added to the book.

| Version | 2009–15 | 2016–21 | 2022–now | Full 2009–now |
|---|---|---|---|---|
| v2 (all parts) | −0.31 | −0.49 | −1.19 | −0.59 |
| Equal dollars (no β) | −0.45 | 0.14 | −0.32 | −0.20 |
| No half-life filter | −0.31 | −0.49 | −1.19 | −0.59 (identical: the filter never binds) |
| Fixed 30-day hold | −0.22 | −0.38 | −1.36 | −0.55 |
| No correlation exit | −0.27 | −0.65 | −0.89 | −0.55 |
| Stocks only | −0.69 | −0.21 | 0.11 | −0.30 |
| ETFs only | 0.20 | −1.17 | −0.89 | −0.55 |

(Sharpe after all costs.) Correlation to the BTC book −0.01; to SPY 0.14.

**Why:** before costs v2 has Sharpe 0.29 (0.51 / 0.43 / −0.28 by period): a small edge that has gone
since 2022. Of 1,246 trades only 197 converged (avg +2.8%, 13 days); 362 hit the 4σ stop (−1.6%),
618 ran out of time or broke correlation (+0.3%). Average gross gain per trade 0.12% vs ~0.4% round-trip
cost. Pairs chosen on last year's data often stop co-moving the next year. This matches the decline
documented by Do & Faff (2010).

The correlation safety exit (Henry's idea) helped in 2016–21 (−0.65 → −0.49) but hurt since 2022.
