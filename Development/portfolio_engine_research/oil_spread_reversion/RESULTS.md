# Oil spread research

In-sample 2012-01-01 → 2021-12-31 (grid of 360 combos per instrument); out-of-sample 2022-01-01 → today, run once on the selected params.

## futures ['CL=F', 'BZ=F']: in-sample

Combos with ≥30 trades: 360 · share with Sharpe > 0: 66% · median Sharpe 0.17

Median in-sample Sharpe by each parameter value (all other params pooled):

- **lookback**: 20: -0.13, 30: 0.07, 40: 0.15, 60: 0.33, 90: 0.22
- **z_in**: 1.5: -0.11, 2.0: 0.15, 2.5: 0.37
- **z_out**: 0.0: 0.22, 0.5: 0.15
- **z_stop**: 3.5: 0.18, 99.0: 0.17
- **max_hold**: 15: 0.11, 30: 0.18, 60: 0.20
- **corr_min**: -1.0: 0.17, 0.7: 0.17

Top 8 by robust (neighbourhood-median) Sharpe:

|   lookback |   z_in |   z_out |   z_stop |   max_hold |   corr_min |   Sharpe |   CAGR % |   Max DD % |   Trades |   Win Rate % |   Robust Sharpe |
|-----------:|-------:|--------:|---------:|-----------:|-----------:|---------:|---------:|-----------:|---------:|-------------:|----------------:|
|         60 |    2.5 |     0   |      3.5 |         60 |       -1   |     0.71 |     7.26 |     -13.64 |       78 |        52.56 |            0.65 |
|         40 |    2.5 |     0.5 |      3.5 |         60 |       -1   |     0.67 |     9.83 |     -23.26 |       86 |        51.16 |            0.64 |
|         60 |    2.5 |     0   |      3.5 |         60 |        0.7 |     0.68 |     7.03 |     -13.68 |       74 |        52.7  |            0.63 |
|         60 |    2.5 |     0   |      3.5 |         30 |       -1   |     0.65 |     6.79 |     -14.16 |       78 |        52.56 |            0.61 |
|         40 |    2.5 |     0.5 |      3.5 |         60 |        0.7 |     0.66 |     9.56 |     -23.14 |       84 |        51.19 |            0.61 |
|         60 |    2.5 |     0   |      3.5 |         30 |        0.7 |     0.63 |     6.56 |     -14.19 |       74 |        52.7  |            0.6  |
|         40 |    2.5 |     0   |      3.5 |         60 |       -1   |     0.65 |    10.07 |     -22.83 |       78 |        51.28 |            0.6  |
|         40 |    2.5 |     0   |      3.5 |         60 |        0.7 |     0.62 |     8.66 |     -21.58 |       76 |        51.32 |            0.58 |

**Selected:** `{'lookback': 60, 'z_in': np.float64(2.5), 'z_out': np.float64(0.0), 'z_stop': np.float64(3.5), 'max_hold': 60, 'corr_min': np.float64(-1.0)}`

## etf ['USO', 'BNO']: in-sample

Combos with ≥30 trades: 360 · share with Sharpe > 0: 8% · median Sharpe -0.59

Median in-sample Sharpe by each parameter value (all other params pooled):

- **lookback**: 20: -0.56, 30: -0.73, 40: -0.56, 60: -0.58, 90: -0.50
- **z_in**: 1.5: -0.79, 2.0: -0.57, 2.5: -0.31
- **z_out**: 0.0: -0.59, 0.5: -0.59
- **z_stop**: 3.5: -0.56, 99.0: -0.61
- **max_hold**: 15: -0.65, 30: -0.62, 60: -0.55
- **corr_min**: -1.0: -0.59, 0.7: -0.59

Top 8 by robust (neighbourhood-median) Sharpe:

|   lookback |   z_in |   z_out |   z_stop |   max_hold |   corr_min |   Sharpe |   CAGR % |   Max DD % |   Trades |   Win Rate % |   Robust Sharpe |
|-----------:|-------:|--------:|---------:|-----------:|-----------:|---------:|---------:|-----------:|---------:|-------------:|----------------:|
|         40 |    2.5 |     0.5 |     99   |         60 |       -1   |     0.59 |     6.48 |     -26.37 |       62 |        50    |            0.39 |
|         40 |    2.5 |     0.5 |     99   |         60 |        0.7 |     0.59 |     6.48 |     -26.37 |       62 |        50    |            0.39 |
|         40 |    2.5 |     0.5 |      3.5 |         60 |       -1   |     0.44 |     3.79 |     -17.04 |       62 |        48.39 |            0.34 |
|         40 |    2.5 |     0.5 |      3.5 |         60 |        0.7 |     0.44 |     3.79 |     -17.04 |       62 |        48.39 |            0.34 |
|         30 |    2.5 |     0.5 |      3.5 |         60 |        0.7 |     0.36 |     2.31 |     -18.37 |       70 |        50    |            0.32 |
|         30 |    2.5 |     0.5 |      3.5 |         60 |       -1   |     0.36 |     2.31 |     -18.37 |       70 |        50    |            0.32 |
|         40 |    2.5 |     0.5 |      3.5 |         30 |       -1   |     0.31 |     2.22 |     -15.94 |       62 |        50    |            0.3  |
|         40 |    2.5 |     0.5 |      3.5 |         30 |        0.7 |     0.31 |     2.22 |     -15.94 |       62 |        50    |            0.3  |

**Selected:** `{'lookback': 40, 'z_in': np.float64(2.5), 'z_out': np.float64(0.5), 'z_stop': np.float64(99.0), 'max_hold': 60, 'corr_min': np.float64(-1.0)}`

## Out-of-sample (selected params, run once)

|                            |   Total Return % |   CAGR % |   Ann. Vol % |     Sharpe |   Max DD % |   Trades |   Win Rate % |   Avg Win % |   Avg Loss % |   Exposure % |   Costs Paid |
|:---------------------------|-----------------:|---------:|-------------:|-----------:|-----------:|---------:|-------------:|------------:|-------------:|-------------:|-------------:|
| futures IS                 |         101.515  |  7.26338 |     10.8087  |   0.70687  |  -13.6425  |       78 |      52.5641 |     9.10935 |     -7.56208 |      31.4629 |      3727.75 |
| futures OOS                |          13.5947 |  2.69612 |      5.97509 |   0.475473 |   -4.15421 |       38 |      60.5263 |     5.22075 |     -5.6825  |      31.7842 |      1584.01 |
| futures OOS costs×2        |          11.887  |  2.37196 |      5.97987 |   0.422116 |   -4.27244 |       38 |      60.5263 |     5.14274 |     -5.75995 |      31.7842 |      3144.03 |
| futures OOS costs×3        |          10.2041 |  2.04865 |      5.98671 |   0.36871  |   -4.39058 |       38 |      60.5263 |     5.06473 |     -5.83738 |      31.7842 |      4680.4  |
| etf IS                     |          87.338  |  6.48324 |     11.7467  |   0.59436  |  -26.3671  |       62 |      50      |     7.99613 |     -9.95721 |      26.7779 |      6843.97 |
| etf OOS                    |         -11.8848 | -2.6106  |      6.15716 |  -0.400549 |  -18.7128  |       40 |      47.5    |     8.22544 |     -7.80588 |      36.3561 |      3768.23 |
| etf OOS costs×2            |         -15.4065 | -3.43758 |      6.19338 |  -0.536008 |  -21.5782  |       40 |      45      |     8.46468 |     -7.6501  |      36.3561 |      7387.52 |
| etf OOS costs×3            |         -18.7912 | -4.25844 |      6.24333 |  -0.668427 |  -24.3458  |       40 |      42.5    |     8.7532  |     -7.52349 |      36.3561 |     10863.6  |
| original CorrReversion OOS |           0      |  0       |      0       | nan        |    0       |        0 |     nan      |   nan       |    nan       |       0      |         0    |

## OOS by year (return %)

|   Date |   futures |   etf |
|-------:|----------:|------:|
|   2022 |      2.53 | -6.44 |
|   2023 |     -0.95 | -2.92 |
|   2024 |     -0.08 | -0.01 |
|   2025 |      1.88 |  0.19 |
|   2026 |      9.88 | -3.15 |

## Signal state today

- **futures**: z = 0.80, 30d corr = 0.79, position: SHORT spread (short Brent / long WTI)
- **etf**: z = 0.68, 30d corr = 0.93, position: flat
## Verdict (2026-10-19)

**No tradeable edge in fading the Brent–WTI spread on daily continuous data.**

1. **Futures (CL=F/BZ=F) results are mostly a roll artefact.** Yahoo's front-month
   series roll on different dates (Brent ~1st of month, WTI ~20th). This creates
   jumps in the spread that "mean-revert" when the other leg rolls, but a real
   position never sees them. Big daily spread moves cluster on days 1–5 (35% of
   them vs ~16% expected). Days 1–5 alone produced ~$49k of the $101k in-sample
   P&L and *more than all* of the out-of-sample P&L ($16.4k of $13.6k).
2. **ETFs (USO/BNO) give honest P&L, and the fade loses.** In-sample median Sharpe
   −0.59 across 360 combos (8% positive); the "best" combo then lost 12% out of
   sample.
3. **Following the spread instead of fading it** (ETFs, same grid): ~0 Sharpe
   in-sample, −0.84 median out-of-sample. No edge either way.

`OilSpreadReversion` stays in the repo as research code. It is **not** recommended
as a live sleeve. Proper testing needs same-delivery-month contract data
(the actual tradeable Brent–WTI spread), not rolled front-month series.
