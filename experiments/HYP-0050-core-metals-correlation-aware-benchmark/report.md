# Core Metals Correlation-Aware Portfolio Benchmark

Objective: test whether covariance-aware allocation reduces drawdowns versus the
1/N and 30-day inverse-volatility baselines.

Construction:

- Source: HYP-0041 raw 5-minute continuous futures close-to-close log returns.
- Evaluation starts at the first available bar after `2021-01-01 00:00:00+00:00`.
- Weights rebalance daily and are applied to same-day 5-minute returns.
- Covariance is estimated from the prior 30 calendar days of 5-minute returns.
- Tested long-only equal-risk-contribution portfolios with covariance shrinkage
to diagonal of 0%, 25%, and 50%.
- Also tested long-only minimum-variance portfolios with 25% and 50% shrinkage.
- No transaction costs are charged in this benchmark.

## Metrics

| variant             | start_ts                  | end_ts                    |   nobs_5m |   years |   cum_log_return |   total_return_pct |   cagr |   annual_log_return |   annual_vol |   sharpe_0rf |   max_drawdown |
|:--------------------|:--------------------------|:--------------------------|----------:|--------:|-----------------:|-------------------:|-------:|--------------------:|-------------:|-------------:|---------------:|
| MINVAR_30D_SHRINK25 | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.8498 |           133.9198 | 0.1683 |              0.1556 |       0.1634 |       0.9519 |        -0.2422 |
| MINVAR_30D_SHRINK50 | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.8022 |           123.0494 | 0.1582 |              0.1469 |       0.1671 |       0.8788 |        -0.2513 |
| INV_VOL_30D_DAILY   | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.5923 |            80.8059 | 0.1145 |              0.1084 |       0.2169 |       0.4999 |        -0.3051 |
| ERC_30D_SHRINK50    | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.5816 |            78.8901 | 0.1124 |              0.1065 |       0.2166 |       0.4917 |        -0.3024 |
| ERC_30D_SHRINK25    | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.5779 |            78.2361 | 0.1116 |              0.1058 |       0.2165 |       0.4887 |        -0.3016 |
| ERC_30D_SHRINK0     | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.5749 |            77.6999 | 0.1110 |              0.1053 |       0.2165 |       0.4862 |        -0.3010 |
| EW_1N               | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.4064 |            50.1439 | 0.0772 |              0.0744 |       0.2561 |       0.2906 |        -0.3474 |

## Best Variant: `MINVAR_30D_SHRINK25`

### Best Weight Summary

| variant             | root   |   mean_weight |   median_weight |   min_weight |   p10_weight |   p90_weight |   max_weight |
|:--------------------|:-------|--------------:|----------------:|-------------:|-------------:|-------------:|-------------:|
| MINVAR_30D_SHRINK25 | GC     |        0.7317 |          0.7506 |       0.2764 |       0.5503 |       0.8960 |       0.9849 |
| MINVAR_30D_SHRINK25 | SI     |        0.0057 |          0.0000 |       0.0000 |       0.0000 |       0.0000 |       0.1653 |
| MINVAR_30D_SHRINK25 | HG     |        0.2159 |          0.1839 |       0.0086 |       0.0695 |       0.3821 |       0.7199 |
| MINVAR_30D_SHRINK25 | PL     |        0.0254 |          0.0000 |       0.0000 |       0.0000 |       0.0794 |       0.3527 |
| MINVAR_30D_SHRINK25 | PA     |        0.0212 |          0.0060 |       0.0000 |       0.0000 |       0.0764 |       0.1570 |

### Best Turnover Summary

| variant             |   cum_turnover |   annual_turnover |   mean_5m_turnover |   p95_5m_turnover |   max_5m_turnover |
|:--------------------|---------------:|------------------:|-------------------:|------------------:|------------------:|
| MINVAR_30D_SHRINK25 |      29.160866 |          5.338748 |           0.000075 |          0.000000 |          0.519048 |

## Optimization Diagnostics

| variant             |   success_rate |   median_avg_pairwise_corr |   median_condition_number |   mean_window_obs |
|:--------------------|---------------:|---------------------------:|--------------------------:|------------------:|
| ERC_30D_SHRINK0     |       1.000000 |                   0.426674 |                 40.023640 |       5820.118737 |
| ERC_30D_SHRINK25    |       1.000000 |                   0.320005 |                 21.993669 |       5820.118737 |
| ERC_30D_SHRINK50    |       1.000000 |                   0.213337 |                 15.192813 |       5820.118737 |
| MINVAR_30D_SHRINK25 |       1.000000 |                   0.320005 |                 21.993669 |       5820.118737 |
| MINVAR_30D_SHRINK50 |       1.000000 |                   0.213337 |                 15.192813 |       5820.118737 |

## Input Span

- start: `2021-01-03 23:00:00+00:00`
- end: `2026-06-21 23:55:00+00:00`
- rows: `387503`
- lookback: `30D`
- minimum lookback rows: `1000`

## Files

- `correlation_aware_cum_log_returns_2021.png`
- `correlation_aware_drawdowns_2021.png`
- `correlation_aware_metric_bars.png`
- `best_correlation_aware_weights_2021.png`
- `core_metals_5m_correlation_aware_benchmark.parquet`
- `daily_rebalance_weights.csv`
- `benchmark_metrics.csv`
- `weights_summary.csv`
- `turnover_summary.csv`
- `rebalance_diagnostics.csv`