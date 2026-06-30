# Core Metals PCA Residual Rolling Correlations

Input: `/home/famadeo/quant-lab/experiments/HYP-0042-core-metals-robust-ewma-pca/robust_ewma_pca_residuals.parquet`.

Method:

- Residuals are from the robust EWMA PCA run after removing PC1-PC2.
- Pairwise correlations use a `30D` rolling window.
- A correlation is emitted only with at least `240` paired observations.
- Plotted values are daily last observations for readability.

## Pair Summary

| pair   | first_ts                  | last_ts                   |   observations |   median_corr |   p10_corr |   p90_corr |   latest_corr |   latest_paired_obs |
|:-------|:--------------------------|:--------------------------|---------------:|--------------:|-----------:|-----------:|--------------:|--------------------:|
| GC-HG  | 2016-07-26 05:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          57332 |       -0.3670 |    -0.6003 |     0.6635 |       -0.7411 |            455.0000 |
| GC-PA  | 2016-07-28 13:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          53744 |        0.1881 |    -0.4106 |     0.6005 |       -0.3775 |            408.0000 |
| GC-PL  | 2016-07-26 19:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          57520 |       -0.3566 |    -0.5415 |    -0.1180 |       -0.1943 |            455.0000 |
| GC-SI  | 2016-07-26 03:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          56330 |       -0.2598 |    -0.6998 |     0.0485 |        0.0225 |            456.0000 |
| HG-PA  | 2016-07-28 14:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          56031 |       -0.5373 |    -0.8654 |     0.6783 |        0.7294 |            408.0000 |
| HG-PL  | 2016-07-26 22:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          58177 |       -0.2682 |    -0.4055 |    -0.0929 |       -0.0166 |            454.0000 |
| PL-PA  | 2016-07-29 05:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          56640 |       -0.4622 |    -0.8840 |    -0.1421 |       -0.6780 |            408.0000 |
| SI-HG  | 2016-07-26 05:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          57318 |       -0.1730 |    -0.4857 |     0.3717 |       -0.6271 |            455.0000 |
| SI-PA  | 2016-07-28 13:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          54530 |        0.0209 |    -0.4153 |     0.4197 |       -0.3654 |            408.0000 |
| SI-PL  | 2016-07-26 19:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          57514 |       -0.3935 |    -0.5300 |    -0.1737 |       -0.1840 |            455.0000 |

## Latest Matrix

|    |      GC |      SI |      HG |      PL |      PA |
|:---|--------:|--------:|--------:|--------:|--------:|
| GC |  1.0000 |  0.0225 | -0.7411 | -0.1943 | -0.3775 |
| SI |  0.0225 |  1.0000 | -0.6271 | -0.1840 | -0.3654 |
| HG | -0.7411 | -0.6271 |  1.0000 | -0.0166 |  0.7294 |
| PL | -0.1943 | -0.1840 | -0.0166 |  1.0000 | -0.6780 |
| PA | -0.3775 | -0.3654 |  0.7294 | -0.6780 |  1.0000 |

## Files

- `residual_rolling_correlations.parquet`
- `residual_rolling_correlations_daily.csv`
- `residual_rolling_correlation_summary.csv`
- `residual_rolling_pairwise_correlations.png`
- `latest_residual_correlation_heatmap.png`