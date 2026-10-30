# Multi-asset trend following on ETFs: results (2026-10-30)

**Verdict: failed the pre-declared bar. Not added to portfolio-engine.**

Strategy (`multi_asset_trend.py`): 18 ETFs (equities SPY IWM EFA EEM EWJ VNQ · bonds IEF TLT TIP LQD ·
commodities GLD SLV DBC USO DBA · FX UUP FXE FXY). Monthly: signal = average sign of 1/3/12-month return,
weight = signal × 40% / (60-day vol × N), long/short, gross capped at 100% (no leverage).
Costs 5 + 5 bp per side, 0.5%/yr borrow on shorts.

Bar (fixed before results): Sharpe ≥ 0.5 over 2008–now, positive in each sub-period, corr to BTC sleeves < 0.3,
and doesn't lower the book's Sharpe.

| Period | Sharpe | CAGR | Max DD |
|---|---|---|---|
| 2008–15 | 0.29 | 1.9% | −14% |
| 2016–21 | 0.19 | 0.8% | −10% |
| 2022–now | 0.04 | 0.0% | −21% |
| Full 2008–now | **0.20** | 1.1% | −21% |

- Benchmark 60/40 SPY/IEF: Sharpe 0.78 (full period).
- Positive in crashes: +11.3% in 2008, +5.4% in 2022. Correlation to SPY −0.28, to the BTC book 0.06.
- Robustness: 2× costs 0.04, 3× costs −0.11; 12m-only 0.21, 3m-only 0.19, 1m-only −0.10.
- Long-only variant: 0.35 / 0.66 / 0.67, full 0.53. It clears 0.5 but trails plain 60/40, and was chosen after
  seeing results, so it would need paper trading as its test.
- Book (BTC OBV + AC, inverse-vol, 50% cap): adding trend cuts max DD since 2022 from −25% to −13%, but Sharpe
  is 1.61 → 1.56 from 2016 and 1.21 → 1.22 from 2022.

Run: copy `multi_asset_trend.py` into `portfolio-engine/strategies/`, then from the portfolio-engine root
`python <path>/research_trend.py`.
