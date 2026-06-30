# Core Metals 1/N Equal-Weight Benchmark

Baseline built from the HYP-0041 raw 5-minute close-to-close continuous futures
log-return panel.

Construction:

- Start timestamp: `2021-01-01 00:00:00+00:00`.
- Asset cumulative returns are simple cumulative sums of 5-minute log returns.
- The 1/N portfolio is rebalanced every 5-minute bar: convert each asset log
return to a simple return, average the five simple returns, then convert the
portfolio bar return back to log return for cumulative log wealth.
- The first in-sample bar is set to zero so the cumulative series starts at zero.

## Metrics

| asset   | start_ts                  | end_ts                    |   nobs_5m |   years |   cum_log_return |   total_return_pct |    cagr |   annual_log_return |   annual_vol |   sharpe_0rf |   max_drawdown |
|:--------|:--------------------------|:--------------------------|----------:|--------:|-----------------:|-------------------:|--------:|--------------------:|-------------:|-------------:|---------------:|
| GC      | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.5495 |            73.2385 |  0.1058 |              0.1006 |       0.1779 |       0.5656 |        -0.2888 |
| SI      | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.6758 |            96.5671 |  0.1317 |              0.1237 |       0.3727 |       0.3320 |        -0.4985 |
| HG      | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.4283 |            53.4654 |  0.0816 |              0.0784 |       0.2773 |       0.2828 |        -0.3821 |
| PL      | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.2907 |            33.7312 |  0.0547 |              0.0532 |       0.3676 |       0.1448 |        -0.4429 |
| PA      | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |          -0.8140 |           -55.6928 | -0.1385 |             -0.1490 |       0.5250 |      -0.2839 |        -0.7726 |
| EW_1N   | 2021-01-03 23:00:00+00:00 | 2026-06-21 23:55:00+00:00 |    387503 |  5.4621 |           0.4064 |            50.1439 |  0.0772 |              0.0744 |       0.2561 |       0.2906 |        -0.3474 |

## Input Span

- start: `2021-01-03 23:00:00+00:00`
- end: `2026-06-21 23:55:00+00:00`
- rows: `387503`

## Files

- `core_metals_1n_cum_log_returns_2021.png`
- `core_metals_asset_cum_log_return_bars_2021.png`
- `core_metals_5m_equal_weight_benchmark.parquet`
- `core_metals_5m_equal_weight_benchmark.csv.gz`
- `core_metals_equal_weight_benchmark_daily.csv`
- `benchmark_metrics.csv`