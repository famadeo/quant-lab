# HYP-0026 Metals Rates + USD Beta

Completed at `2026-06-26T01:02:24.184783+00:00`.

## Definition

- `rates_price`: equal-weight daily log return of `ZT`, `ZF`, `ZN`, `ZB`. Positive means Treasury futures prices up / yields lower.
- `usd`: negative equal-weight daily log return of `6E`, `6B`, `6J`, `6A`, `6C`. Positive means broad USD stronger versus those FX futures.
- Regression: `metal_return = alpha + beta_rates_price * rates_price + beta_usd * usd + residual`.
- Beta columns in bp are metal return bp for a +1% move in the factor.

## Coverage

- Factor span: `2010-06-07` to `2024-11-29`.
- Trade-era overlap used here: `2023-06-22` to `2024-11-29`.
- Rates/USD factor correlation: `-0.157`.

## Trade-Era Overlap Betas

| root   |   nobs |   metal_bp_per_1pct_usd |   metal_bp_per_1pct_rates_price |     r2 |   beta_usd_t |   beta_rates_price_t |
|:-------|-------:|------------------------:|--------------------------------:|-------:|-------------:|---------------------:|
| GC     |    449 |               -111.4674 |                         16.1407 | 0.2515 |      -9.4526 |               1.3048 |
| HG     |    449 |               -203.7065 |                        -53.3305 | 0.2571 |     -11.6921 |              -2.9181 |
| PA     |    449 |               -292.8026 |                        -26.9073 | 0.1912 |      -8.9872 |              -0.7873 |
| PL     |    449 |               -221.7827 |                        -31.9841 | 0.2472 |     -10.8527 |              -1.4920 |
| SI     |    449 |               -258.4040 |                        -25.0273 | 0.2541 |     -10.8141 |              -0.9985 |

## Latest Rolling 252d Betas

| root   | date                      |   metal_bp_per_1pct_usd |   metal_bp_per_1pct_rates_price |     r2 |
|:-------|:--------------------------|------------------------:|--------------------------------:|-------:|
| GC     | 2024-11-29 00:00:00+00:00 |               -146.3540 |                         -8.0702 | 0.2705 |
| HG     | 2024-11-29 00:00:00+00:00 |               -243.9736 |                        -80.6183 | 0.2602 |
| PA     | 2024-11-29 00:00:00+00:00 |               -320.5778 |                        -48.3090 | 0.1837 |
| PL     | 2024-11-29 00:00:00+00:00 |               -232.2531 |                        -62.3747 | 0.2069 |
| SI     | 2024-11-29 00:00:00+00:00 |               -316.9470 |                       -101.9561 | 0.2410 |

## Files

- `factor_returns_daily.csv`
- `full_sample_betas.csv`
- `rolling_252d_betas.csv`
- `latest_rolling_252d_betas.csv`
- `macro_fitted_residual_returns.parquet`
- `latest_beta_bars.png`
- `rolling_betas_and_r2.png`
