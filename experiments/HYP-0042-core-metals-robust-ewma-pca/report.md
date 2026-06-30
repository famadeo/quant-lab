# Core Metals Robust EWMA Rolling PCA

Completed at `2026-06-29T01:34:37.518998+00:00`.

## Data

- Input: `/home/famadeo/quant-lab/experiments/HYP-0041-core-metals-5m-log-returns/core_metals_5m_log_returns_wide.parquet` and `/home/famadeo/quant-lab/experiments/HYP-0041-core-metals-5m-log-returns/core_metals_5m_log_returns_long.parquet`.
- Assets: `GC, SI, HG, PL, PA`.
- Span: `2016-06-22 00:00:00+00:00` to `2026-06-21 23:55:00+00:00`.
- Rows: `708,026` five-minute timestamps.
- PCA diagnostics: `58,691` emitted rows from `2016-07-11 17:55:00+00:00` to `2026-06-21 23:55:00+00:00`.

## Method

- Use observed 5-minute returns only for PCA estimation; stale aligned bars are masked.
- Standardize each asset by lagged EWMA volatility.
- EWMA volatility half-life: `864` bars, minimum `288` observations.
- Clip standardized returns to `+/-6` before correlation estimation.
- Estimate EWMA correlation with half-life `2880` bars.
- Require `1440` pair observations before emitting PCA state.
- Update EWMA every 5-minute bar; emit PCA diagnostics every `12` bars.
- Shrink correlation matrix toward identity with alpha `0.1`.
- Project to nearest positive semi-definite correlation matrix before eigendecomposition.
- Align eigenvector signs to the prior timestamp.
- Residual dislocation removes the first `2` PCs.

## Latest State

- Latest PCA timestamp: `2026-06-21 23:55:00+00:00`.
- PC1 explained variance: `0.6609`.
- PC2 explained variance: `0.1195`.
- PC3 explained variance: `0.1063`.
- Effective rank: `2.9753`.

### Latest Loadings

| ts                        | component   | root   |   loading |   explained_variance |   eigenvalue |   effective_rank |
|:--------------------------|:------------|:-------|----------:|---------------------:|-------------:|-----------------:|
| 2026-06-21 23:55:00+00:00 | PC1         | GC     |   -0.4619 |               0.6609 |       3.3047 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC1         | SI     |   -0.4769 |               0.6609 |       3.3047 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC1         | HG     |   -0.3990 |               0.6609 |       3.3047 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC1         | PL     |   -0.4689 |               0.6609 |       3.3047 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC1         | PA     |   -0.4245 |               0.6609 |       3.3047 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC2         | GC     |    0.1817 |               0.1195 |       0.5973 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC2         | SI     |    0.1634 |               0.1195 |       0.5973 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC2         | HG     |    0.6526 |               0.1195 |       0.5973 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC2         | PL     |   -0.3187 |               0.1195 |       0.5973 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC2         | PA     |   -0.6425 |               0.1195 |       0.5973 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC3         | GC     |    0.5362 |               0.1063 |       0.5316 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC3         | SI     |    0.4001 |               0.1063 |       0.5316 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC3         | HG     |   -0.6413 |               0.1063 |       0.5316 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC3         | PL     |   -0.0529 |               0.1063 |       0.5316 |           2.9753 |
| 2026-06-21 23:55:00+00:00 | PC3         | PA     |   -0.3719 |               0.1063 |       0.5316 |           2.9753 |

## Residual Norm Summary

|       |      value |
|:------|-----------:|
| count | 58058.0000 |
| mean  |     0.9900 |
| std   |     0.6787 |
| min   |     0.0000 |
| 50%   |     0.8234 |
| 90%   |     1.8435 |
| 95%   |     2.2990 |
| 99%   |     3.4279 |
| max   |     9.9774 |

## Top Residual Dislocations

| ts                        |   observed_count |   residual_norm |   max_abs_residual |   residual_GC |   residual_SI |   residual_HG |   residual_PL |   residual_PA |
|:--------------------------|-----------------:|----------------:|-------------------:|--------------:|--------------:|--------------:|--------------:|--------------:|
| 2025-01-15 13:30:00+00:00 |                5 |          9.9774 |             8.9421 |        1.1852 |        0.9239 |       -8.9421 |        2.2372 |        3.5106 |
| 2021-11-28 23:00:00+00:00 |                5 |          7.1116 |             4.9976 |       -2.9326 |        1.9124 |        1.4359 |        3.3584 |       -4.9976 |
| 2022-10-10 01:00:00+00:00 |                4 |          7.0778 |             4.6953 |       -0.0675 |       -4.2444 |        4.6953 |       -0.1049 |      nan      |
| 2021-09-13 14:45:00+00:00 |                5 |          6.4085 |             4.5655 |       -1.4334 |        1.6883 |        3.9018 |        0.3099 |       -4.5655 |
| 2018-12-06 18:15:00+00:00 |                4 |          6.3888 |             5.1557 |       -0.2469 |      nan      |        5.1557 |       -2.4047 |       -0.4776 |
| 2017-04-06 09:20:00+00:00 |                4 |          6.3217 |             5.3056 |        1.6489 |        0.9218 |        0.5031 |      nan      |       -5.3056 |
| 2020-04-21 09:35:00+00:00 |                5 |          6.2660 |             4.5891 |       -1.8506 |        1.4758 |        4.5891 |       -0.7365 |       -3.4724 |
| 2024-02-11 23:00:00+00:00 |                5 |          6.2217 |             5.5115 |        1.8961 |        0.9208 |       -5.5115 |        0.0197 |        1.9720 |
| 2020-08-19 01:15:00+00:00 |                4 |          6.1512 |             5.1789 |       -0.9756 |       -1.1339 |        5.1789 |       -1.1005 |      nan      |
| 2018-11-30 14:00:00+00:00 |                4 |          6.0411 |             4.0253 |        2.3980 |       -4.0253 |       -0.5707 |        2.6299 |      nan      |

## Files

- `robust_ewma_pca_state.parquet`
- `robust_ewma_pca_residuals.parquet`
- `latest_loadings.csv`
- `top_residual_dislocations.csv`
- `explained_variance_effective_rank.png`
- `pc1_loadings_over_time.png`
- `pc2_pc3_loadings_over_time.png`
- `pc_scores_cumulative.png`
- `residual_dislocation_norm.png`
- `asset_residual_abs_p95.png`
- `latest_loadings_bar.png`