# Core Metals PC1-PC2 Residual Cumulative Log Returns

Standardized residual input: `/home/famadeo/quant-lab/experiments/HYP-0042-core-metals-robust-ewma-pca/robust_ewma_pca_residuals.parquet`.

Method:

- Residuals are from the robust EWMA PCA run after removing PC1-PC2.
- Convert standardized residuals back into return units by multiplying by each asset's lagged EWMA 5-minute volatility.
- EWMA volatility half-life: `864` bars.
- EWMA volatility minimum observations: `288` bars.
- Cumulative paths fill missing residual returns with zero.
- Plots use daily last cumulative values for readability.

Caveat: these are residual returns at the emitted PCA diagnostic timestamps, not every raw 5-minute bar.

## Summary

| root   |   nobs |   final_cum_residual_log_return |   mean_residual_return_bps |   std_residual_return_bps |   mean_abs_residual_return_bps |   p95_abs_residual_return_bps |
|:-------|-------:|--------------------------------:|---------------------------:|--------------------------:|-------------------------------:|------------------------------:|
| GC     |  56201 |                        0.022445 |                   0.003994 |                  2.889071 |                       1.887363 |                      5.694243 |
| SI     |  56699 |                       -0.285995 |                  -0.050441 |                  5.616831 |                       3.532328 |                     10.737719 |
| HG     |  57369 |                       -0.108391 |                  -0.018894 |                  5.941199 |                       3.783602 |                     12.176839 |
| PL     |  57140 |                        0.196724 |                   0.034428 |                  7.044696 |                       4.672291 |                     14.058964 |
| PA     |  42764 |                        0.302671 |                   0.070777 |                 12.061399 |                       6.989500 |                     21.980638 |

## Coverage

- First residual timestamp: `2016-07-11 17:55:00+00:00`.
- Last residual timestamp: `2026-06-21 23:55:00+00:00`.
- Emitted rows: `58,691`.

## Files

- `pc12_residual_log_returns.parquet`
- `pc12_residual_cumulative_log_returns_daily.csv`
- `pc12_residual_cumulative_log_returns_summary.csv`
- `pc12_residual_cumulative_log_returns_overlay.png`
- `pc12_residual_cumulative_log_returns_panels.png`