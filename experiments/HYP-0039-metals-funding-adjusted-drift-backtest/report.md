# HYP-0039 Metals Funding-Adjusted Drift Backtest

## Strategy

For each metal, tenor, and rolling window:

`pressure = trailing_realized_log_return - trailing_funding_paid`

Positive pressure means the long side has been winning after carry. Negative pressure means the short side has been winning after carry.

The backtest uses a state machine:

- Enter long when pressure score is above the entry threshold.
- Enter short when pressure score is below the negative entry threshold.
- Hold until the score crosses the exit threshold.
- Apply the position to the next hourly return after funding.

## Implementation

- Source accounting panel: `HYP-0037` hourly realized return minus funding paid.
- Score methods: raw pressure divided by trailing realized volatility, and rolling pressure z-score.
- Funding materiality filter: `abs(trailing_funding_paid) / trailing_vol`.
- Portfolio construction: equal 20% capital sleeve per metal, inactive sleeve in cash.
- Costs: per-side MBP1 cost estimates multiplied by turnover and cost multiplier.
- Train/test split: chronological 70/30 with a 7-day embargo before test.

## Cost Assumptions

| root   |   per_side_cost_bps |
|:-------|--------------------:|
| GC     |              0.5508 |
| SI     |              1.8695 |
| HG     |              0.8004 |
| PL     |              2.5632 |
| PA     |              5.5939 |

## Best Train-Selected Variants at 1x Costs

| variant                                                          | split   |   net_return |   gross_excess_return |   cost_return |   sharpe |   tstat |   max_drawdown |   active_fraction |   turnover |   gross_to_cost |
|:-----------------------------------------------------------------|:--------|-------------:|----------------------:|--------------:|---------:|--------:|---------------:|------------------:|-----------:|----------------:|
| target1m_lb2w_pressure_vol_scaled_entry0p5_exit0_filt0p02_costx1 | full    |      -0.1000 |               -0.0278 |        0.0722 |  -0.0995 | -0.3148 |        -0.5475 |            0.8457 |   363.2000 |         -0.3844 |
| target1m_lb2w_pressure_vol_scaled_entry0p5_exit0_filt0p02_costx1 | test    |      -0.2886 |               -0.2577 |        0.0309 |  -0.6378 | -1.1015 |        -0.5150 |            0.9743 |   140.4000 |         -8.3266 |
| target1m_lb2w_pressure_vol_scaled_entry0p5_exit0_filt0p02_costx1 | train   |       0.1779 |                0.2190 |        0.0411 |   0.3749 |  0.9917 |        -0.1117 |            0.7906 |   222.2000 |          5.3285 |
| target1m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1   | full    |      -0.0186 |                0.0212 |        0.0398 |  -0.0208 | -0.0657 |        -0.4964 |            0.7613 |   192.0000 |          0.5320 |
| target1m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1   | test    |      -0.2331 |               -0.2165 |        0.0166 |  -0.5676 | -0.9803 |        -0.4849 |            0.8662 |    72.8000 |        -13.0668 |
| target1m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1   | train   |       0.2074 |                0.2305 |        0.0231 |   0.5083 |  1.3447 |        -0.0676 |            0.7163 |   118.4000 |          9.9955 |
| target1m_lb2w_pressure_z_entry0p5_exit0_filt0p02_costx1          | full    |      -0.0447 |                0.0300 |        0.0747 |  -0.0438 | -0.1386 |        -0.5087 |            0.8391 |   374.6000 |          0.4017 |
| target1m_lb2w_pressure_z_entry0p5_exit0_filt0p02_costx1          | test    |      -0.2675 |               -0.2363 |        0.0312 |  -0.5787 | -0.9995 |        -0.4939 |            0.9849 |   144.6000 |         -7.5798 |
| target1m_lb2w_pressure_z_entry0p5_exit0_filt0p02_costx1          | train   |       0.2117 |                0.2550 |        0.0433 |   0.4464 |  1.1807 |        -0.0943 |            0.7766 |   229.4000 |          5.8922 |
| target1m_lb2w_pressure_z_entry1_exit0_filt0_costx1               | full    |      -0.0700 |               -0.0150 |        0.0550 |  -0.0440 | -0.1391 |        -0.6313 |            0.9514 |   250.6000 |         -0.2724 |
| target1m_lb2w_pressure_z_entry1_exit0_filt0_costx1               | test    |      -0.2523 |               -0.2325 |        0.0198 |  -0.4084 | -0.7053 |        -0.5052 |            0.9523 |    86.2000 |        -11.7348 |
| target1m_lb2w_pressure_z_entry1_exit0_filt0_costx1               | train   |       0.1764 |                0.2115 |        0.0350 |   0.1887 |  0.4992 |        -0.2848 |            0.9508 |   163.8000 |          6.0396 |
| target1m_lb2w_pressure_z_entry1_exit0_filt0p02_costx1            | full    |       0.0354 |                0.0765 |        0.0411 |   0.0396 |  0.1254 |        -0.4000 |            0.7770 |   206.0000 |          1.8595 |
| target1m_lb2w_pressure_z_entry1_exit0_filt0p02_costx1            | test    |      -0.1866 |               -0.1699 |        0.0167 |  -0.4626 | -0.7988 |        -0.4000 |            0.9127 |    79.4000 |        -10.1729 |
| target1m_lb2w_pressure_z_entry1_exit0_filt0p02_costx1            | train   |       0.2189 |                0.2433 |        0.0244 |   0.5238 |  1.3854 |        -0.1009 |            0.7186 |   126.2000 |          9.9877 |
| target3m_lb2w_pressure_vol_scaled_entry0p5_exit0_filt0p02_costx1 | full    |      -0.0938 |               -0.0228 |        0.0710 |  -0.0957 | -0.3025 |        -0.5024 |            0.8213 |   352.0000 |         -0.3213 |
| target3m_lb2w_pressure_vol_scaled_entry0p5_exit0_filt0p02_costx1 | test    |      -0.3100 |               -0.2794 |        0.0306 |  -0.7033 | -1.2145 |        -0.4787 |            0.9746 |   138.0000 |         -9.1303 |
| target3m_lb2w_pressure_vol_scaled_entry0p5_exit0_filt0p02_costx1 | train   |       0.2052 |                0.2454 |        0.0402 |   0.4417 |  1.1683 |        -0.1148 |            0.7555 |   213.6000 |          6.1023 |
| target3m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1   | full    |      -0.0607 |               -0.0224 |        0.0383 |  -0.0698 | -0.2208 |        -0.4745 |            0.7347 |   187.0000 |         -0.5852 |
| target3m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1   | test    |      -0.3048 |               -0.2888 |        0.0161 |  -0.7688 | -1.3277 |        -0.4745 |            0.8631 |    72.6000 |        -17.9810 |
| target3m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1   | train   |       0.2352 |                0.2572 |        0.0220 |   0.5910 |  1.5633 |        -0.0685 |            0.6796 |   113.6000 |         11.6933 |
| target3m_lb2w_pressure_z_entry0p5_exit0_filt0p02_costx1          | full    |      -0.0378 |                0.0369 |        0.0746 |  -0.0380 | -0.1202 |        -0.4606 |            0.8168 |   369.8000 |          0.4939 |
| target3m_lb2w_pressure_z_entry0p5_exit0_filt0p02_costx1          | test    |      -0.2696 |               -0.2377 |        0.0319 |  -0.5985 | -1.0336 |        -0.4349 |            0.9770 |   147.0000 |         -7.4440 |
| target3m_lb2w_pressure_z_entry0p5_exit0_filt0p02_costx1          | train   |       0.2208 |                0.2633 |        0.0425 |   0.4780 |  1.2644 |        -0.1022 |            0.7480 |   222.4000 |          6.1889 |
| target6m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1   | full    |       0.1608 |                0.1775 |        0.0167 |   0.2485 |  0.7856 |        -0.3134 |            0.6851 |   123.2000 |         10.6442 |
| target6m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1   | test    |      -0.0550 |               -0.0475 |        0.0075 |  -0.1904 | -0.3287 |        -0.3125 |            0.8579 |    52.6000 |         -6.3289 |
| target6m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1   | train   |       0.2073 |                0.2163 |        0.0090 |   0.6667 |  1.7636 |        -0.0904 |            0.6106 |    69.8000 |         24.0316 |
| target6m_lb3d_pressure_z_entry1p5_exit0p5_filt0_costx1           | full    |       0.2148 |                0.2724 |        0.0576 |   0.2424 |  0.7665 |        -0.2075 |            0.5427 |   403.8000 |          4.7302 |
| target6m_lb3d_pressure_z_entry1p5_exit0p5_filt0_costx1           | test    |       0.0388 |                0.0622 |        0.0234 |   0.1068 |  0.1844 |        -0.2043 |            0.6060 |   145.6000 |          2.6589 |
| target6m_lb3d_pressure_z_entry1p5_exit0p5_filt0_costx1           | train   |       0.1762 |                0.2102 |        0.0340 |   0.3600 |  0.9524 |        -0.1890 |            0.5154 |   257.2000 |          6.1740 |

## Top Full-Sample Variants at 1x Costs

| variant                                                           | split   |   net_return |   gross_excess_return |   cost_return |   sharpe |   tstat |   max_drawdown |   active_fraction |   turnover |   gross_to_cost |
|:------------------------------------------------------------------|:--------|-------------:|----------------------:|--------------:|---------:|--------:|---------------:|------------------:|-----------:|----------------:|
| target6m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0_costx1     | full    |       0.3224 |                0.4560 |        0.1336 |   0.3237 |  1.0234 |        -0.2232 |            0.7917 |   962.6000 |          3.4139 |
| target6m_lb3d_pressure_z_entry1p5_exit0_filt0_costx1              | full    |       0.3063 |                0.3565 |        0.0502 |   0.3207 |  1.0140 |        -0.2023 |            0.6638 |   355.0000 |          7.0966 |
| target6m_lb3d_pressure_z_entry1p5_exit0p25_filt0_costx1           | full    |       0.2906 |                0.3433 |        0.0527 |   0.3163 |  1.0001 |        -0.1991 |            0.6002 |   371.8000 |          6.5112 |
| target1m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0_costx1     | full    |       0.2723 |                0.5499 |        0.2775 |   0.2031 |  0.6422 |        -0.3835 |            0.8687 |  1307.6000 |          1.9812 |
| target6m_lb3d_pressure_z_entry1_exit0p5_filt0_costx1              | full    |       0.2710 |                0.3932 |        0.1222 |   0.2639 |  0.8346 |        -0.1856 |            0.7446 |   874.8000 |          3.2174 |
| target1m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0p02_costx1  | full    |       0.2274 |                0.3076 |        0.0802 |   0.5967 |  1.8866 |        -0.0868 |            0.4008 |   574.8000 |          3.8359 |
| target6m_lb3d_pressure_vol_scaled_entry1_exit0p25_filt0_costx1    | full    |       0.2267 |                0.3388 |        0.1120 |   0.2135 |  0.6751 |        -0.2492 |            0.8556 |   801.2000 |          3.0235 |
| target1m_lb3d_pressure_z_entry1_exit0p25_filt0_costx1             | full    |       0.2243 |                0.4411 |        0.2168 |   0.1556 |  0.4919 |        -0.2631 |            0.8836 |  1004.2000 |          2.0345 |
| target1m_lb3d_pressure_vol_scaled_entry1_exit0p25_filt0p02_costx1 | full    |       0.2236 |                0.2984 |        0.0749 |   0.5581 |  1.7647 |        -0.0869 |            0.4367 |   524.8000 |          3.9863 |
| target3m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0_costx1     | full    |       0.2217 |                0.5000 |        0.2783 |   0.1653 |  0.5228 |        -0.3941 |            0.8704 |  1316.4000 |          1.7965 |
| target1m_lb3d_pressure_z_entry1p5_exit0_filt0_costx1              | full    |       0.2189 |                0.3290 |        0.1100 |   0.1720 |  0.5437 |        -0.2742 |            0.7477 |   503.6000 |          2.9896 |
| target6m_lb3d_pressure_z_entry1_exit0p25_filt0_costx1             | full    |       0.2166 |                0.3195 |        0.1029 |   0.2013 |  0.6365 |        -0.2086 |            0.8092 |   732.0000 |          3.1051 |
| target6m_lb3d_pressure_z_entry1p5_exit0p5_filt0_costx1            | full    |       0.2148 |                0.2724 |        0.0576 |   0.2424 |  0.7665 |        -0.2075 |            0.5427 |   403.8000 |          4.7302 |
| target1m_lb3d_pressure_z_entry1_exit0_filt0_costx1                | full    |       0.2137 |                0.4056 |        0.1920 |   0.1427 |  0.4513 |        -0.3203 |            0.9257 |   891.8000 |          2.1132 |
| target3m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0p02_costx1  | full    |       0.2131 |                0.2897 |        0.0766 |   0.5923 |  1.8729 |        -0.0868 |            0.3822 |   547.6000 |          3.7823 |
| target6m_lb3d_pressure_z_entry1_exit0_filt0_costx1                | full    |       0.2094 |                0.2994 |        0.0900 |   0.1873 |  0.5921 |        -0.2275 |            0.8650 |   640.6000 |          3.3269 |
| target1m_lb1w_pressure_z_entry0p5_exit0_filt0p02_costx1           | full    |       0.2045 |                0.2997 |        0.0952 |   0.2761 |  0.8730 |        -0.2205 |            0.7774 |   544.2000 |          3.1491 |
| target3m_lb3d_pressure_z_entry1p5_exit0_filt0_costx1              | full    |       0.2016 |                0.3128 |        0.1112 |   0.1576 |  0.4982 |        -0.2448 |            0.7530 |   512.0000 |          2.8138 |
| target3m_lb3d_pressure_z_entry1_exit0_filt0_costx1                | full    |       0.1959 |                0.3888 |        0.1929 |   0.1308 |  0.4135 |        -0.2791 |            0.9292 |   900.6000 |          2.0152 |
| target3m_lb3d_pressure_vol_scaled_entry1_exit0p25_filt0p02_costx1 | full    |       0.1951 |                0.2669 |        0.0718 |   0.5162 |  1.6324 |        -0.0893 |            0.4156 |   504.8000 |          3.7191 |

## Top Test Variants at 1x Costs

| variant                                                             | split   |   net_return |   gross_excess_return |   cost_return |   sharpe |   tstat |   max_drawdown |   active_fraction |   turnover |   gross_to_cost |
|:--------------------------------------------------------------------|:--------|-------------:|----------------------:|--------------:|---------:|--------:|---------------:|------------------:|-----------:|----------------:|
| target1m_lb3d_pressure_z_entry1_exit0_filt0_costx1                  | test    |       0.2096 |                0.2720 |        0.0624 |   0.3670 |  0.6338 |        -0.2408 |            0.9434 |   279.6000 |          4.3603 |
| target1m_lb3d_pressure_vol_scaled_entry1_exit0_filt0p02_costx1      | test    |       0.1970 |                0.2294 |        0.0325 |   1.0130 |  1.7493 |        -0.0825 |            0.7228 |   217.8000 |          7.0679 |
| target1m_lb3d_pressure_vol_scaled_entry0p5_exit0_filt0p02_costx1    | test    |       0.1940 |                0.2505 |        0.0565 |   0.8707 |  1.5036 |        -0.0830 |            0.8526 |   373.4000 |          4.4371 |
| target1m_lb3d_pressure_vol_scaled_entry1_exit0p25_filt0p02_costx1   | test    |       0.1897 |                0.2230 |        0.0332 |   1.0143 |  1.7516 |        -0.0869 |            0.6848 |   227.4000 |          6.7076 |
| target3m_lb3d_pressure_vol_scaled_entry0p5_exit0_filt0p02_costx1    | test    |       0.1883 |                0.2410 |        0.0527 |   0.8853 |  1.5289 |        -0.0621 |            0.8462 |   356.8000 |          4.5695 |
| target3m_lb3d_pressure_z_entry1_exit0_filt0_costx1                  | test    |       0.1854 |                0.2483 |        0.0630 |   0.3255 |  0.5621 |        -0.2535 |            0.9461 |   282.0000 |          3.9437 |
| target1m_lb3d_pressure_z_entry0p5_exit0_filt0p02_costx1             | test    |       0.1780 |                0.2284 |        0.0504 |   0.8185 |  1.4135 |        -0.0952 |            0.8401 |   342.2000 |          4.5316 |
| target3m_lb3d_pressure_z_entry0p5_exit0_filt0p02_costx1             | test    |       0.1706 |                0.2187 |        0.0480 |   0.8184 |  1.4134 |        -0.0790 |            0.8363 |   332.6000 |          4.5519 |
| target1m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0p02_costx1    | test    |       0.1553 |                0.1918 |        0.0365 |   0.8708 |  1.5039 |        -0.0868 |            0.6354 |   257.8000 |          5.2544 |
| target1m_lb3d_pressure_vol_scaled_entry0p5_exit0p25_filt0p02_costx1 | test    |       0.1546 |                0.2230 |        0.0684 |   0.7183 |  1.2405 |        -0.1050 |            0.8200 |   468.2000 |          3.2594 |
| target3m_lb3d_pressure_vol_scaled_entry0p5_exit0_filt0_costx1       | test    |       0.1545 |                0.2699 |        0.1154 |   0.2410 |  0.4163 |        -0.3240 |            0.9964 |   508.8000 |          2.3385 |
| target3m_lb3d_pressure_vol_scaled_entry1_exit0_filt0p02_costx1      | test    |       0.1542 |                0.1857 |        0.0315 |   0.8289 |  1.4314 |        -0.0675 |            0.7243 |   214.2000 |          5.8948 |
| target6m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0_costx1       | test    |       0.1484 |                0.1970 |        0.0486 |   0.3748 |  0.6472 |        -0.1677 |            0.8218 |   319.4000 |          4.0542 |
| target1m_lb3d_pressure_z_entry0p5_exit0p25_filt0p02_costx1          | test    |       0.1472 |                0.2072 |        0.0599 |   0.7094 |  1.2251 |        -0.1145 |            0.7920 |   425.8000 |          3.4561 |
| target1m_lb3d_pressure_z_entry1p5_exit0_filt0_costx1                | test    |       0.1452 |                0.1817 |        0.0365 |   0.3005 |  0.5189 |        -0.2416 |            0.7603 |   157.8000 |          4.9829 |
| target1m_lb3d_pressure_z_entry1_exit0p25_filt0p02_costx1            | test    |       0.1451 |                0.1719 |        0.0269 |   0.8682 |  1.4993 |        -0.0937 |            0.6397 |   198.8000 |          6.4008 |
| target6m_lb3d_pressure_z_entry1p5_exit0_filt0_costx1                | test    |       0.1434 |                0.1632 |        0.0199 |   0.3665 |  0.6329 |        -0.2023 |            0.7224 |   125.2000 |          8.2200 |
| target1m_lb3d_pressure_z_entry1_exit0_filt0p02_costx1               | test    |       0.1430 |                0.1691 |        0.0261 |   0.8206 |  1.4171 |        -0.0890 |            0.6847 |   189.6000 |          6.4798 |
| target3m_lb3d_pressure_vol_scaled_entry1_exit0p25_filt0p02_costx1   | test    |       0.1430 |                0.1757 |        0.0327 |   0.8019 |  1.3849 |        -0.0893 |            0.6799 |   227.0000 |          5.3761 |
| target3m_lb3d_pressure_vol_scaled_entry0p5_exit0p25_filt0p02_costx1 | test    |       0.1422 |                0.2064 |        0.0642 |   0.6918 |  1.1947 |        -0.0912 |            0.8107 |   448.4000 |          3.2142 |

## Strict Train/Test Robustness at 1x Costs

| variant                                                           |   full_net |   train_net |   test_net |   full_sharpe |   train_sharpe |   test_sharpe |   train_gross_to_cost |   test_gross_to_cost |   active_fraction |   full_max_drawdown |
|:------------------------------------------------------------------|-----------:|------------:|-----------:|--------------:|---------------:|--------------:|----------------------:|---------------------:|------------------:|--------------------:|
| target1m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0p02_costx1  |     0.2274 |      0.0763 |     0.1553 |        0.5967 |         0.4649 |        0.8708 |                2.7546 |               5.2544 |            0.4008 |             -0.0868 |
| target3m_lb3d_pressure_vol_scaled_entry1_exit0p25_filt0p02_costx1 |     0.1951 |      0.0567 |     0.1430 |        0.5162 |         0.3573 |        0.8019 |                2.4577 |               5.3761 |            0.4156 |             -0.0893 |
| target3m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0p02_costx1  |     0.2131 |      0.0852 |     0.1320 |        0.5923 |         0.5681 |        0.7757 |                3.0976 |               4.6876 |            0.3822 |             -0.0868 |
| target1m_lb1w_pressure_z_entry0p5_exit0_filt0p02_costx1           |     0.2045 |      0.0644 |     0.1386 |        0.2761 |         0.1841 |        0.4155 |                2.1288 |               4.6597 |            0.7774 |             -0.2205 |
| target6m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0p02_costx1  |     0.1095 |      0.0553 |     0.0532 |        0.3673 |         0.4194 |        0.3850 |                3.3624 |               2.8843 |            0.3579 |             -0.0839 |
| target6m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0_costx1     |     0.3224 |      0.1703 |     0.1484 |        0.3237 |         0.2981 |        0.3748 |                3.0068 |               4.0542 |            0.7917 |             -0.2232 |
| target6m_lb3d_pressure_z_entry1p5_exit0p25_filt0_costx1           |     0.2906 |      0.1519 |     0.1391 |        0.3163 |         0.2985 |        0.3700 |                5.7918 |               7.6599 |            0.6002 |             -0.1991 |
| target6m_lb3d_pressure_z_entry1p5_exit0_filt0_costx1              |     0.3063 |      0.1598 |     0.1434 |        0.3207 |         0.3025 |        0.3665 |                6.2839 |               8.2200 |            0.6638 |             -0.2023 |
| target6m_lb3d_pressure_z_entry1_exit0p25_filt0_costx1             |     0.2166 |      0.0773 |     0.1331 |        0.2013 |         0.1294 |        0.3028 |                2.2190 |               4.3745 |            0.8092 |             -0.2086 |
| target6m_lb3d_pressure_z_entry1_exit0_filt0_costx1                |     0.2094 |      0.0620 |     0.1376 |        0.1873 |         0.0994 |        0.3025 |                2.1042 |               5.0729 |            0.8650 |             -0.2275 |
| target1m_lb3d_pressure_z_entry1p5_exit0_filt0_costx1              |     0.2189 |      0.0772 |     0.1452 |        0.1720 |         0.1009 |        0.3005 |                2.0525 |               4.9829 |            0.7477 |             -0.2742 |
| target1m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0_costx1     |     0.2723 |      0.1507 |     0.1216 |        0.2031 |         0.1860 |        0.2403 |                1.7940 |               2.3912 |            0.8687 |             -0.3835 |
| target6m_lb3d_pressure_vol_scaled_entry1_exit0p25_filt0_costx1    |     0.2267 |      0.1202 |     0.1014 |        0.2135 |         0.1982 |        0.2395 |                2.6591 |               3.5692 |            0.8556 |             -0.2492 |
| target1m_lb3d_pressure_z_entry1_exit0p25_filt0_costx1             |     0.2243 |      0.1015 |     0.1289 |        0.1556 |         0.1176 |        0.2345 |                1.6985 |               2.8203 |            0.8836 |             -0.2631 |
| target3m_lb3d_pressure_z_entry1p5_exit0_filt0_costx1              |     0.2016 |      0.0936 |     0.1131 |        0.1576 |         0.1220 |        0.2321 |                2.2699 |               4.0347 |            0.7530 |             -0.2448 |
| target6m_lb3d_pressure_z_entry1_exit0p5_filt0_costx1              |     0.2710 |      0.1694 |     0.0951 |        0.2639 |         0.2975 |        0.2266 |                3.2643 |               3.0094 |            0.7446 |             -0.1856 |
| target6m_lb3d_pressure_vol_scaled_entry1p5_exit0p5_filt0_costx1   |     0.1555 |      0.0799 |     0.0706 |        0.1912 |         0.1683 |        0.2221 |                2.9814 |               4.1004 |            0.6041 |             -0.2138 |
| target3m_lb3d_pressure_vol_scaled_entry1_exit0p5_filt0_costx1     |     0.2217 |      0.1164 |     0.1063 |        0.1653 |         0.1428 |        0.2114 |                1.6108 |               2.2165 |            0.8704 |             -0.3941 |
| target6m_lb3d_pressure_vol_scaled_entry0p5_exit0_filt0_costx1     |     0.1705 |      0.1129 |     0.0552 |        0.1364 |         0.1582 |        0.1108 |                2.0381 |               1.9069 |            0.9819 |             -0.2417 |

## Root Breakdown For Best Train-Selected Variant

| variant                                                        | root   |   observations |   active_bars |   gross_excess_return |   price_return |   carry_return |   cost_return |   net_return |   turnover |   mean_score |   mean_pressure_bp |   target_months | lookback   | score_method        |   cost_multiplier |
|:---------------------------------------------------------------|:-------|---------------:|--------------:|----------------------:|---------------:|---------------:|--------------:|-------------:|-----------:|-------------:|-------------------:|----------------:|:-----------|:--------------------|------------------:|
| target3m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1 | GC     |          58381 |         29429 |                0.0744 |         0.0782 |        -0.0038 |        0.0022 |       0.0722 |    39.8000 |       0.0895 |            30.9690 |               3 | 2w         | pressure_vol_scaled |            1.0000 |
| target3m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1 | PA     |          49577 |         10297 |                0.0777 |         0.0707 |         0.0071 |        0.0142 |       0.0635 |    25.4000 |       0.1334 |            61.9328 |               3 | 2w         | pressure_vol_scaled |            1.0000 |
| target3m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1 | PL     |          58856 |         16019 |                0.0141 |         0.0131 |         0.0010 |        0.0110 |       0.0031 |    43.0000 |      -0.0192 |             3.0334 |               3 | 2w         | pressure_vol_scaled |            1.0000 |
| target3m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1 | SI     |          58194 |         25057 |               -0.0257 |        -0.0233 |        -0.0024 |        0.0079 |      -0.0336 |    42.4000 |       0.0319 |            37.1328 |               3 | 2w         | pressure_vol_scaled |            1.0000 |
| target3m_lb2w_pressure_vol_scaled_entry1_exit0_filt0p02_costx1 | HG     |          57690 |         17514 |               -0.1629 |        -0.1584 |        -0.0045 |        0.0029 |      -0.1658 |    36.4000 |       0.1012 |            37.2703 |               3 | 2w         | pressure_vol_scaled |            1.0000 |

## Pressure State Diagnostic, 3M Target, 1W Lookback

|   target_months | lookback   | root   | pressure_state      |   observations |   next_excess_bp |   next_return_bp |   next_funding_paid_bp |   positive_excess_fraction |
|----------------:|:-----------|:-------|:--------------------|---------------:|-----------------:|-----------------:|-----------------------:|---------------------------:|
|               3 | 1w         | GC     | longs_pay_and_lose  |          27317 |           0.0902 |           0.1441 |                 0.0539 |                     0.4941 |
|               3 | 1w         | GC     | longs_pay_and_win   |          30868 |           0.0807 |           0.1351 |                 0.0544 |                     0.4995 |
|               3 | 1w         | GC     | shorts_pay_and_lose |             28 |          -3.5903 |          -3.5842 |                 0.0061 |                     0.5357 |
|               3 | 1w         | HG     | longs_pay_and_lose  |          23809 |           0.1563 |           0.2033 |                 0.0470 |                     0.4854 |
|               3 | 1w         | HG     | longs_pay_and_win   |          26095 |           0.0653 |           0.1101 |                 0.0448 |                     0.4825 |
|               3 | 1w         | HG     | shorts_pay_and_lose |           4230 |          -0.7732 |          -0.7921 |                -0.0190 |                     0.5021 |
|               3 | 1w         | HG     | shorts_pay_and_win  |           3388 |           1.2284 |           1.2187 |                -0.0097 |                     0.5215 |
|               3 | 1w         | PA     | longs_pay_and_lose  |          15669 |           0.2117 |           0.2608 |                 0.0491 |                     0.4967 |
|               3 | 1w         | PA     | longs_pay_and_win   |          14554 |          -0.3305 |          -0.2896 |                 0.0409 |                     0.4835 |
|               3 | 1w         | PA     | shorts_pay_and_lose |          12088 |           0.4801 |           0.4440 |                -0.0360 |                     0.5220 |
|               3 | 1w         | PA     | shorts_pay_and_win  |           7098 |           0.5863 |           0.5591 |                -0.0272 |                     0.5218 |
|               3 | 1w         | PL     | longs_pay_and_lose  |          28601 |           0.0391 |           0.0795 |                 0.0404 |                     0.4885 |
|               3 | 1w         | PL     | longs_pay_and_win   |          27153 |          -0.0999 |          -0.0620 |                 0.0378 |                     0.5006 |
|               3 | 1w         | PL     | shorts_pay_and_lose |           1832 |          -0.7079 |          -0.7203 |                -0.0124 |                     0.4869 |
|               3 | 1w         | PL     | shorts_pay_and_win  |           1102 |           2.2158 |           2.2141 |                -0.0017 |                     0.5245 |
|               3 | 1w         | SI     | longs_pay_and_lose  |          27936 |           0.0356 |           0.0931 |                 0.0575 |                     0.4774 |
|               3 | 1w         | SI     | longs_pay_and_win   |          30090 |           0.1504 |           0.2078 |                 0.0573 |                     0.4849 |

## Files

- `variant_metrics.csv`
- `root_variant_metrics.csv`
- `pressure_state_diagnostics.csv`
- `top_train_selected_returns.parquet`
- `top_train_selected_equity.png`
- `top_full_sample_variant_metrics.png`

## Caveats

- This uses front-futures proxy funding, not true spot/cash funding.
- Hourly rebalancing is intentionally conservative on cost sensitivity, but it may overstate turnover relative to a production implementation with execution bands.
- The strategy is directional, not beta-neutral. Positive results can still be trend beta unless they survive hedging or cross-sectional construction.