# Core Metals PC1-PC2 Residual Return Minus Carry Cost

Definition:

`after_carry = residual_log_return - integrated_residual_carry_cost`

Method:

- Residual returns: `/home/famadeo/quant-lab/experiments/HYP-0042-core-metals-robust-ewma-pca/pc12_residual_log_returns.parquet`.
- Residual carry rates: `/home/famadeo/quant-lab/experiments/HYP-0042-core-metals-robust-ewma-pca/pc12_residual_carry_cost_pct_ann.parquet`.
- Carry is annualized percent paid by a long residual basket.
- Carry cost is integrated over elapsed clock time using the lagged carry rate.
- Missing residual returns are treated as zero, matching the residual cumulative plot.
- Rolling windows require a minimum number of valid emitted PCA observations.

Caveat: residual returns are only available at emitted PCA diagnostic timestamps, not every raw 5-minute bar. Carry accrues over clock time.

## Full Sample Cumulative

| root   |   nobs |   cum_residual_log_return |   cum_carry_cost_log |   cum_after_carry_log_return |   cum_after_carry_bp |
|:-------|-------:|--------------------------:|---------------------:|-----------------------------:|---------------------:|
| GC     |  57512 |                  0.024550 |             0.089719 |                    -0.065169 |          -651.685407 |
| HG     |  57512 |                 -0.111640 |             0.024963 |                    -0.136603 |         -1366.028379 |
| PA     |  57512 |                  0.270483 |            -0.021643 |                     0.292126 |          2921.255581 |
| PL     |  57512 |                  0.263515 |            -0.040603 |                     0.304118 |          3041.182407 |
| SI     |  57512 |                 -0.314105 |            -0.047777 |                    -0.266327 |         -2663.270816 |

## Rolling Window Summary

| window   | root   |   nobs |   mean_after_carry_bp |   median_after_carry_bp |   p10_after_carry_bp |   p90_after_carry_bp |   latest_after_carry_bp |   positive_fraction |   mean_residual_bp |   mean_carry_cost_bp |
|:---------|:-------|-------:|----------------------:|------------------------:|---------------------:|---------------------:|------------------------:|--------------------:|-------------------:|---------------------:|
| 120D     | GC     |  57731 |              -22.6638 |                -18.4832 |            -172.5146 |             106.7859 |                 46.9341 |              0.4336 |             7.1700 |              29.8338 |
| 120D     | HG     |  57731 |              -46.6642 |                -45.8497 |            -298.2547 |             226.1040 |                 16.3537 |              0.3844 |           -38.9935 |               7.6707 |
| 120D     | PA     |  57731 |              101.5283 |                 66.3627 |            -455.6493 |             570.2382 |               -221.9787 |              0.5951 |            93.7649 |              -7.7634 |
| 120D     | PL     |  57731 |               99.1176 |                 77.1077 |            -281.3168 |             492.5289 |                382.2408 |              0.5996 |            85.7733 |             -13.3443 |
| 120D     | SI     |  57731 |              -87.3943 |                -49.0622 |            -381.2945 |             188.7506 |               -408.1151 |              0.3862 |          -102.4842 |             -15.0899 |
| 1D       | GC     |  51848 |               -0.1226 |                 -0.1292 |             -14.7417 |              14.2576 |                  1.7061 |              0.4937 |             0.1833 |               0.3059 |
| 1D       | HG     |  51848 |               -0.1247 |                 -0.2254 |             -31.2950 |              31.1272 |                 -3.7874 |              0.4926 |            -0.0401 |               0.0846 |
| 1D       | PA     |  51848 |                0.7779 |                  0.1645 |             -42.8644 |              46.6359 |                -34.5768 |              0.5028 |             0.7029 |              -0.0750 |
| 1D       | PL     |  51848 |                0.9370 |                  0.2168 |             -35.0680 |              37.8782 |                 15.4827 |              0.5045 |             0.7987 |              -0.1383 |
| 1D       | SI     |  51848 |               -1.1965 |                 -0.6299 |             -27.9205 |              26.1844 |                 -0.7614 |              0.4855 |            -1.3579 |              -0.1614 |
| 20D      | GC     |  57753 |               -4.1989 |                 -2.6933 |             -62.4584 |              49.7881 |                 63.9589 |              0.4738 |             0.8462 |               5.0451 |
| 20D      | HG     |  57753 |               -7.0896 |                 -7.3693 |            -122.9970 |             115.1639 |                -38.7981 |              0.4541 |            -5.7280 |               1.3615 |
| 20D      | PA     |  57753 |               16.2443 |                 -1.1680 |            -157.4973 |             213.3864 |                 29.8728 |              0.4955 |            15.0160 |              -1.2283 |
| 20D      | PL     |  57753 |               16.4228 |                 17.4192 |            -139.3960 |             179.3744 |               -102.3905 |              0.5635 |            14.1443 |              -2.2785 |
| 20D      | SI     |  57753 |              -14.5980 |                 -9.0642 |            -123.9734 |              98.2516 |                 30.4778 |              0.4545 |           -17.2510 |              -2.6531 |
| 252D     | GC     |  56691 |              -47.3749 |                -24.6392 |            -319.8313 |             155.4489 |                -30.9914 |              0.4299 |            15.0020 |              62.3769 |
| 252D     | HG     |  56691 |             -109.9520 |                -91.1307 |            -474.5044 |             199.4621 |                361.8247 |              0.3337 |           -95.0207 |              14.9313 |
| 252D     | PA     |  56691 |              206.3803 |                155.9075 |            -634.4398 |            1238.7604 |                255.7262 |              0.6194 |           188.7642 |             -17.6161 |
| 252D     | PL     |  56691 |              206.9641 |                 90.3859 |            -317.8687 |             980.0653 |               -142.9395 |              0.6267 |           179.2798 |             -27.6843 |
| 252D     | SI     |  56691 |             -144.3879 |               -153.9567 |            -529.7260 |             262.9909 |              -1002.8182 |              0.3146 |          -174.2103 |             -29.8225 |
| 5D       | GC     |  56338 |               -1.5119 |                 -1.2735 |             -29.5073 |              26.9568 |                 21.6340 |              0.4757 |            -0.1379 |               1.3740 |
| 5D       | HG     |  56338 |               -1.1075 |                 -2.2354 |             -58.2726 |              58.1571 |                -23.2668 |              0.4687 |            -0.7391 |               0.3685 |
| 5D       | PA     |  56338 |                3.3019 |                  0.8877 |             -85.3556 |              95.0864 |                -77.9600 |              0.5071 |             2.9735 |              -0.3284 |
| 5D       | PL     |  56338 |                3.5707 |                  2.6435 |             -64.9246 |              74.8983 |                 16.8960 |              0.5204 |             2.9520 |              -0.6187 |
| 5D       | SI     |  56338 |               -2.8604 |                 -1.5059 |             -52.4861 |              51.5684 |                 17.8588 |              0.4841 |            -3.5865 |              -0.7261 |
| 60D      | GC     |  58211 |              -12.6097 |                -14.5823 |            -113.0825 |              83.2146 |                 -3.1374 |              0.4363 |             2.3663 |              14.9759 |
| 60D      | HG     |  58211 |              -21.9945 |                -17.9336 |            -231.3965 |             172.2177 |                 -3.8394 |              0.4344 |           -18.0313 |               3.9632 |
| 60D      | PA     |  58211 |               50.0512 |                 16.2414 |            -299.3980 |             433.1339 |                 77.5908 |              0.5282 |            46.3090 |              -3.7422 |
| 60D      | PL     |  58211 |               50.8622 |                 34.6861 |            -228.9564 |             367.9133 |                -21.7980 |              0.5723 |            44.1266 |              -6.7356 |
| 60D      | SI     |  58211 |              -44.7170 |                -31.1483 |            -235.3930 |             158.7286 |                -58.6112 |              0.4098 |           -52.4742 |              -7.7572 |

## Files

- `pc12_residual_after_carry_accounting.parquet`
- `pc12_residual_after_carry_rolling.parquet`
- `pc12_residual_after_carry_rolling_daily.csv`
- `pc12_residual_after_carry_full_summary.csv`
- `pc12_residual_after_carry_rolling_summary.csv`
- `pc12_residual_after_carry_cumulative_overlay.png`
- `pc12_residual_after_carry_cumulative_components.png`
- `pc12_residual_after_carry_rolling_windows.png`
- `pc12_residual_after_carry_latest_heatmap.png`