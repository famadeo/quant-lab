# Core Metals Hourly Return Autoregression

Objective: measure how autoregressive hourly log returns are for each core metal.

Construction:

- Source: HYP-0041 raw 5-minute continuous futures close-to-close log returns.
- Evaluation starts at the first available bar after `2021-01-01 00:00:00+00:00`.
- Hourly returns are summed 5-minute log returns by observed hourly bucket.
- `contiguous` statistics only use lag pairs where the timestamp difference is exactly
the stated lag in hours, avoiding Friday-to-Sunday/session-gap lag pairs.
- AR model t-stats use HAC standard errors with up to 24 hourly lags.

## Contiguous-Hour AR(1)

| root   |   nobs |   lag1_coef |   lag1_tstat |   lag1_pvalue |       r2 |   adj_r2 |
|:-------|-------:|------------:|-------------:|--------------:|---------:|---------:|
| GC     |  30899 |   -0.013365 |    -1.342867 |      0.179315 | 0.000188 | 0.000156 |
| SI     |  30899 |   -0.008533 |    -0.699408 |      0.484297 | 0.000076 | 0.000043 |
| HG     |  30899 |   -0.011457 |    -0.659130 |      0.509812 | 0.000135 | 0.000103 |
| PL     |  30899 |   -0.023949 |    -2.709417 |      0.006740 | 0.000592 | 0.000560 |
| PA     |  30899 |   -0.052685 |    -5.562606 |      0.000000 | 0.002842 | 0.002810 |

## Contiguous-Hour AR(6) Summary

| root   |   nobs |   lag1_coef |   lag1_tstat |   lag1_pvalue |       r2 |   adj_r2 |
|:-------|-------:|------------:|-------------:|--------------:|---------:|---------:|
| GC     |  23845 |   -0.006920 |    -0.556935 |      0.577572 | 0.000665 | 0.000414 |
| SI     |  23845 |    0.008904 |     0.588814 |      0.555986 | 0.001554 | 0.001303 |
| HG     |  23845 |   -0.014574 |    -0.725722 |      0.468009 | 0.000436 | 0.000184 |
| PL     |  23845 |   -0.019068 |    -2.027696 |      0.042591 | 0.000729 | 0.000478 |
| PA     |  23845 |   -0.048639 |    -4.617738 |      0.000004 | 0.002806 | 0.002555 |

## Selected Autocorrelations

| root   |         1 |         2 |         3 |         6 |        12 |   24 |
|:-------|----------:|----------:|----------:|----------:|----------:|-----:|
| GC     | -0.013718 | -0.014625 |  0.002518 | -0.001399 | -0.007912 |  nan |
| SI     | -0.008709 | -0.023718 |  0.007298 | -0.006967 |  0.011506 |  nan |
| HG     | -0.011622 | -0.008593 | -0.011275 | -0.000038 | -0.005317 |  nan |
| PL     | -0.024328 | -0.007023 |  0.001402 |  0.016665 | -0.020140 |  nan |
| PA     | -0.053313 | -0.003112 | -0.011359 | -0.008185 |  0.003370 |  nan |

## AR(6) Coefficients

| root   |         1 |         2 |         3 |         4 |         5 |         6 |
|:-------|----------:|----------:|----------:|----------:|----------:|----------:|
| GC     | -0.006920 | -0.013104 |  0.012723 |  0.014689 | -0.006953 | -0.001096 |
| SI     |  0.008904 | -0.019915 |  0.014572 |  0.005357 | -0.026793 | -0.006769 |
| HG     | -0.014574 | -0.011584 | -0.004606 |  0.007967 | -0.002382 | -0.000122 |
| PL     | -0.019068 | -0.007888 |  0.004058 | -0.000009 |  0.003343 |  0.016665 |
| PA     | -0.048639 | -0.007384 | -0.010113 |  0.014973 | -0.000089 | -0.008361 |

## Ljung-Box On Observed-Hour Sequence

| root   | sample       |   lag |    lb_stat |   lb_pvalue |
|:-------|:-------------|------:|-----------:|------------:|
| GC     | all_observed |     1 |   4.875931 |    0.027234 |
| GC     | all_observed |     6 |  12.741588 |    0.047328 |
| GC     | all_observed |    12 |  33.200277 |    0.000901 |
| GC     | all_observed |    24 |  77.253453 |    0.000000 |
| SI     | all_observed |     1 |   2.465833 |    0.116346 |
| SI     | all_observed |     6 |  25.148159 |    0.000321 |
| SI     | all_observed |    12 |  86.118517 |    0.000000 |
| SI     | all_observed |    24 | 195.216487 |    0.000000 |
| HG     | all_observed |     1 |   5.832669 |    0.015731 |
| HG     | all_observed |     6 |  21.927363 |    0.001248 |
| HG     | all_observed |    12 |  35.010099 |    0.000467 |
| HG     | all_observed |    24 |  58.351500 |    0.000109 |
| PL     | all_observed |     1 |  22.081901 |    0.000003 |
| PL     | all_observed |     6 |  27.178353 |    0.000134 |
| PL     | all_observed |    12 |  36.312847 |    0.000289 |
| PL     | all_observed |    24 | 111.983293 |    0.000000 |
| PA     | all_observed |     1 |  92.039538 |    0.000000 |
| PA     | all_observed |     6 | 106.763127 |    0.000000 |
| PA     | all_observed |    12 | 128.667701 |    0.000000 |
| PA     | all_observed |    24 | 179.940246 |    0.000000 |

## Input Span

- hourly start: `2021-01-03 23:00:00+00:00`
- hourly end: `2026-06-21 23:00:00+00:00`
- observed hourly rows: `32311`

## Files

- `core_metals_hourly_log_returns.parquet`
- `core_metals_hourly_log_returns.csv.gz`
- `acf_summary.csv`
- `ar_model_summary.csv`
- `ar_coefficients.csv`
- `ljung_box_summary.csv`
- `rolling_90d_ar1.csv`
- `hourly_autocorrelation_heatmap_contiguous.png`
- `hourly_ar1_coefficients_contiguous.png`
- `rolling_90d_hourly_ar1.png`
- `hourly_autocorrelation_lag_curves.png`