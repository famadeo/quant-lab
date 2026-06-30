# Core Metals 30-Day Inverse-Volatility Benchmark

Baseline built from the HYP-0041 raw 5-minute close-to-close continuous futures
log-return panel.

Construction:

- Evaluation starts at the first available bar after `2021-01-01 00:00:00+00:00`.
- Trailing volatility is the rolling 30-calendar-day standard deviation of each
metal's 5-minute log returns.
- Volatility is shifted by one bar before weight calculation to avoid lookahead.
- Weights are long-only inverse-volatility weights: `w_i ∝ 1 / sigma_i`, normalized
to sum to 1.
- The portfolio is rebalanced every 5-minute bar before costs.
- This is equal standalone-vol weighting, not correlation-aware risk parity.

## Metrics

| asset       | start_ts                  | end_ts                    |   nobs_5m |   years |   cum_log_return |   total_return_pct |    cagr |   annual_log_return |   annual_vol |   sharpe_0rf |   max_drawdown |
|:------------|:--------------------------|:--------------------------|----------:|--------:|-----------------:|-------------------:|--------:|--------------------:|-------------:|-------------:|---------------:|
| GC          | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.5495 |            73.2385 |  0.1058 |              0.1006 |       0.1779 |       0.5656 |        -0.2888 |
| SI          | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.6758 |            96.5671 |  0.1317 |              0.1237 |       0.3727 |       0.3320 |        -0.4985 |
| HG          | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.4283 |            53.4654 |  0.0816 |              0.0784 |       0.2773 |       0.2828 |        -0.3821 |
| PL          | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.2907 |            33.7312 |  0.0547 |              0.0532 |       0.3676 |       0.1448 |        -0.4429 |
| PA          | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |          -0.8140 |           -55.6928 | -0.1385 |             -0.1490 |       0.5250 |      -0.2839 |        -0.7726 |
| EW_1N       | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.4064 |            50.1439 |  0.0772 |              0.0744 |       0.2561 |       0.2906 |        -0.3474 |
| INV_VOL_30D | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.5933 |            81.0004 |  0.1147 |              0.1086 |       0.2168 |       0.5012 |        -0.3059 |

## Weight Summary

| root   |   mean_weight |   median_weight |   min_weight |   p10_weight |   p90_weight |   max_weight |
|:-------|--------------:|----------------:|-------------:|-------------:|-------------:|-------------:|
| GC     |        0.3410 |          0.3426 |       0.2092 |       0.2902 |       0.3871 |       0.4381 |
| SI     |        0.1709 |          0.1713 |       0.1025 |       0.1384 |       0.1962 |       0.2673 |
| HG     |        0.2138 |          0.2085 |       0.0655 |       0.1747 |       0.2692 |       0.3423 |
| PL     |        0.1640 |          0.1645 |       0.0966 |       0.1370 |       0.1881 |       0.2461 |
| PA     |        0.1103 |          0.1069 |       0.0615 |       0.0804 |       0.1458 |       0.1897 |

## Turnover Summary

| portfolio   |   cum_turnover |   annual_turnover |   mean_5m_turnover |   p95_5m_turnover |   max_5m_turnover |
|:------------|---------------:|------------------:|-------------------:|------------------:|------------------:|
| INV_VOL_30D |      39.591684 |          7.248414 |           0.000102 |          0.000321 |          0.192387 |

## Input Span

- start: `2021-01-03 23:00:00+00:00`
- end: `2026-06-21 23:55:00+00:00`
- rows: `387503`
- volatility lookback: `30D`
- min volatility observations: `1000`

## Files

- `core_metals_inverse_vol_vs_1n_cum_log_returns_2021.png`
- `core_metals_inverse_vol_relative_to_1n_2021.png`
- `core_metals_inverse_vol_weights_2021.png`
- `core_metals_inverse_vol_average_weights_2021.png`
- `core_metals_5m_inverse_vol_benchmark.parquet`
- `core_metals_5m_inverse_vol_benchmark.csv.gz`
- `core_metals_inverse_vol_benchmark_daily.csv`
- `benchmark_metrics.csv`
- `weights_summary.csv`
- `turnover_summary.csv`