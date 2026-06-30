# Core Metals PCA Residual Rolling Correlations After PC1-PC3

State input: `/home/famadeo/quant-lab/experiments/HYP-0042-core-metals-robust-ewma-pca/robust_ewma_pca_state.parquet`.
PC1-PC2 residual input: `/home/famadeo/quant-lab/experiments/HYP-0042-core-metals-robust-ewma-pca/robust_ewma_pca_residuals.parquet`.

Method:

- Start from robust EWMA PCA residuals after removing PC1-PC2.
- Subtract the PC3 loading times PC3 score at each emitted PCA timestamp.
- Require at least four observed assets before retaining a PC1-PC3 residual row.
- Pairwise correlations use a `30D` rolling window.
- A correlation is emitted only with at least `240` paired observations.
- Plotted values are daily last observations for readability.

## Pair Summary

| pair   | first_ts                  | last_ts                   |   observations |   median_corr |   p10_corr |   p90_corr |   latest_corr |   latest_paired_obs |
|:-------|:--------------------------|:--------------------------|---------------:|--------------:|-----------:|-----------:|--------------:|--------------------:|
| GC-HG  | 2016-07-26 10:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          56526 |        0.8784 |     0.6921 |     0.9787 |        0.9605 |            454.0000 |
| GC-PA  | 2016-07-28 13:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          53271 |        0.4678 |     0.2945 |     0.6587 |        0.4237 |            408.0000 |
| GC-PL  | 2016-07-26 20:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          56931 |       -0.4479 |    -0.6245 |    -0.1966 |       -0.3042 |            454.0000 |
| GC-SI  | 2016-07-26 09:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          56174 |       -0.5981 |    -0.9344 |    -0.3289 |       -0.8528 |            454.0000 |
| HG-PA  | 2016-07-28 14:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          55711 |        0.7611 |     0.1104 |     0.9412 |        0.2215 |            408.0000 |
| HG-PL  | 2016-07-26 22:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          57996 |       -0.7305 |    -0.9293 |    -0.0015 |       -0.1009 |            454.0000 |
| PL-PA  | 2016-07-29 05:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          56038 |       -0.9961 |    -0.9991 |    -0.9831 |       -0.9898 |            408.0000 |
| SI-HG  | 2016-07-26 10:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          57137 |       -0.2426 |    -0.9429 |     0.1371 |       -0.9276 |            454.0000 |
| SI-PA  | 2016-07-28 13:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          54440 |        0.3919 |     0.0293 |     0.5438 |        0.1021 |            408.0000 |
| SI-PL  | 2016-07-26 20:55:00+00:00 | 2026-06-21 23:55:00+00:00 |          57366 |       -0.4255 |    -0.5611 |    -0.1758 |       -0.2375 |            454.0000 |

## Latest Matrix

|    |      GC |      SI |      HG |      PL |      PA |
|:---|--------:|--------:|--------:|--------:|--------:|
| GC |  1.0000 | -0.8528 |  0.9605 | -0.3042 |  0.4237 |
| SI | -0.8528 |  1.0000 | -0.9276 | -0.2375 |  0.1021 |
| HG |  0.9605 | -0.9276 |  1.0000 | -0.1009 |  0.2215 |
| PL | -0.3042 | -0.2375 | -0.1009 |  1.0000 | -0.9898 |
| PA |  0.4237 |  0.1021 |  0.2215 | -0.9898 |  1.0000 |

## Files

- `robust_ewma_pca_residuals_pc1_pc3.parquet`
- `residual_pc1_pc3_rolling_correlations.parquet`
- `residual_pc1_pc3_rolling_correlations_daily.csv`
- `residual_pc1_pc3_rolling_correlation_summary.csv`
- `residual_pc1_pc3_rolling_pairwise_correlations.png`
- `latest_residual_pc1_pc3_correlation_heatmap.png`