# HYP-0032 Metals Convenience-Yield Robust Extreme Detectors

## Design

- Uses the HYP-0031 synchronized 5-minute curve panel.
- Compares rolling empirical percentile, rolling quantile bands, median/MAD z-score, and robust diagonal curve-state MD.
- Position logic, costs, tenors, volume filters, event exits, and root weights match the prior convenience-yield tests.

## Best Variant

| variant                                               | detector   |   min_volume |   cost_multiplier |   net_return |   cost_return |   sharpe |   tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|:------------------------------------------------------|:-----------|-------------:|------------------:|-------------:|--------------:|---------:|--------:|---------------:|--------------:|--------------:|------------------:|
| rank_pct_target3m_minv10_lb126_entry0p1_both_costx1p0 | rank_pct   |      10.0000 |            1.0000 |       0.0729 |        0.0625 |   2.6333 |  4.5594 |        -0.0094 |           275 |        7.4693 |            0.8699 |

## Best By Detector

| detector      |   target_months |   min_volume |   cost_multiplier | variant                                                     |   net_return |   cost_return |   sharpe |   tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|:--------------|----------------:|-------------:|------------------:|:------------------------------------------------------------|-------------:|--------------:|---------:|--------:|---------------:|--------------:|--------------:|------------------:|
| rank_pct      |               3 |      10.0000 |            1.0000 | rank_pct_target3m_minv10_lb126_entry0p1_both_costx1p0       |       0.0729 |        0.0625 |   2.6333 |  4.5594 |        -0.0094 |           275 |        7.4693 |            0.8699 |
| quantile_band |               3 |      10.0000 |            1.0000 | quantile_band_target3m_minv10_lb126_entry0p1_both_costx1p0  |       0.0721 |        0.0626 |   2.5829 |  4.4722 |        -0.0094 |           277 |        7.4005 |            0.8720 |
| mad_z         |               1 |      10.0000 |            1.0000 | mad_z_target1m_minv10_lb126_entry1p5_both_costx1p0          |       0.0675 |        0.0812 |   2.9208 |  5.0573 |        -0.0053 |           321 |        7.8227 |            0.8544 |
| curve_md      |               1 |      10.0000 |            1.0000 | curve_md_target1m_minv10_lb126_md2p5_comp1p25_both_costx1p0 |       0.0562 |        0.0648 |   2.7144 |  4.6998 |        -0.0045 |           312 |        8.8684 |            0.6777 |

## Volume And Cost Robustness

| detector      |   target_months |   min_volume |   cost_multiplier | variant                                                                     |   net_return |   cost_return |   sharpe |    tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|:--------------|----------------:|-------------:|------------------:|:----------------------------------------------------------------------------|-------------:|--------------:|---------:|---------:|---------------:|--------------:|--------------:|------------------:|
| curve_md      |               1 |      10.0000 |            1.0000 | curve_md_target1m_minv10_lb126_md2p5_comp1p25_both_costx1p0                 |       0.0562 |        0.0648 |   2.7144 |   4.6998 |        -0.0045 |           312 |        8.8684 |            0.6777 |
| curve_md      |               6 |      10.0000 |            3.0000 | curve_md_target6m_minv10_lb126_md3p0_comp1p25_both_costx3p0                 |      -0.0097 |        0.0677 |  -0.4394 |  -0.7602 |        -0.0187 |           144 |        0.3155 |            0.5933 |
| curve_md      |               3 |     500.0000 |            1.0000 | curve_md_target3m_minv500_lb252_md2p5_comp1p25_both_costx1p0                |       0.0147 |        0.0124 |   0.7464 |   1.2912 |        -0.0085 |            75 |        2.1505 |            0.5834 |
| curve_md      |               6 |     500.0000 |            3.0000 | curve_md_target6m_minv500_lb126_md3p0_comp1p25_both_costx3p0                |       0.0066 |        0.0055 |   0.7911 |   1.3617 |        -0.0016 |            19 |        1.1814 |            0.2867 |
| curve_md      |               1 |    1000.0000 |            1.0000 | curve_md_target1m_minv1000_lb126_md3p0_comp1p25_both_costx1p0               |       0.0036 |        0.0045 |   0.2445 |   0.4229 |        -0.0075 |            44 |        2.0351 |            0.2397 |
| curve_md      |               6 |    1000.0000 |            3.0000 | curve_md_target6m_minv1000_lb126_md2p5_comp1p25_both_costx3p0               |      -0.0032 |        0.0019 |  -0.7551 |  -1.2942 |        -0.0042 |            10 |       -1.0387 |            0.3662 |
| mad_z         |               1 |      10.0000 |            1.0000 | mad_z_target1m_minv10_lb126_entry1p5_both_costx1p0                          |       0.0675 |        0.0812 |   2.9208 |   5.0573 |        -0.0053 |           321 |        7.8227 |            0.8544 |
| mad_z         |               3 |      10.0000 |            3.0000 | mad_z_target3m_minv10_lb126_entry2p0_contango_only_costx3p0                 |       0.0049 |        0.0474 |   0.3372 |   0.5839 |        -0.0071 |            70 |        4.8375 |            0.3484 |
| mad_z         |               1 |     500.0000 |            1.0000 | mad_z_target1m_minv500_lb126_entry1p5_both_costx1p0                         |       0.0214 |        0.0241 |   1.1822 |   2.0451 |        -0.0087 |           133 |        3.8387 |            0.8052 |
| mad_z         |               6 |     500.0000 |            3.0000 | mad_z_target6m_minv500_lb252_entry2p0_backwardation_only_costx3p0           |       0.0029 |        0.0028 |   0.3986 |   0.6861 |        -0.0021 |             4 |        0.6507 |            0.3986 |
| mad_z         |               1 |    1000.0000 |            1.0000 | mad_z_target1m_minv1000_lb126_entry1p5_both_costx1p0                        |       0.0158 |        0.0131 |   0.9713 |   1.6802 |        -0.0081 |            90 |        4.1129 |            0.7346 |
| mad_z         |               3 |    1000.0000 |            3.0000 | mad_z_target3m_minv1000_lb126_entry2p0_contango_only_costx3p0               |       0.0005 |        0.0062 |   0.0852 |   0.1474 |        -0.0039 |            14 |        0.6653 |            0.2047 |
| quantile_band |               3 |      10.0000 |            1.0000 | quantile_band_target3m_minv10_lb126_entry0p1_both_costx1p0                  |       0.0721 |        0.0626 |   2.5829 |   4.4722 |        -0.0094 |           277 |        7.4005 |            0.8720 |
| quantile_band |               6 |      10.0000 |            3.0000 | quantile_band_target6m_minv10_lb252_entry0p05_backwardation_only_costx3p0   |       0.0016 |        0.0362 |   0.0809 |   0.1400 |        -0.0108 |            76 |        0.7055 |            0.5689 |
| quantile_band |               1 |     500.0000 |            1.0000 | quantile_band_target1m_minv500_lb126_entry0p1_both_costx1p0                 |       0.0208 |        0.0293 |   1.1266 |   1.9488 |        -0.0081 |           170 |        3.7831 |            0.8424 |
| quantile_band |               6 |     500.0000 |            3.0000 | quantile_band_target6m_minv500_lb126_entry0p1_backwardation_only_costx3p0   |       0.0033 |        0.0028 |   0.4625 |   0.7960 |        -0.0022 |             4 |        0.7635 |            0.3741 |
| quantile_band |               1 |    1000.0000 |            1.0000 | quantile_band_target1m_minv1000_lb126_entry0p1_both_costx1p0                |       0.0184 |        0.0154 |   1.0737 |   1.8573 |        -0.0080 |           116 |        4.8005 |            0.7705 |
| quantile_band |               6 |    1000.0000 |            3.0000 | quantile_band_target6m_minv1000_lb252_entry0p05_backwardation_only_costx3p0 |       0.0000 |        0.0000 | nan      | nan      |         0.0000 |             0 |      nan      |            0.0000 |
| rank_pct      |               3 |      10.0000 |            1.0000 | rank_pct_target3m_minv10_lb126_entry0p1_both_costx1p0                       |       0.0729 |        0.0625 |   2.6333 |   4.5594 |        -0.0094 |           275 |        7.4693 |            0.8699 |
| rank_pct      |               6 |      10.0000 |            3.0000 | rank_pct_target6m_minv10_lb252_entry0p05_backwardation_only_costx3p0        |       0.0015 |        0.0361 |   0.0744 |   0.1288 |        -0.0109 |            76 |        0.6923 |            0.5655 |
| rank_pct      |               1 |     500.0000 |            1.0000 | rank_pct_target1m_minv500_lb126_entry0p1_both_costx1p0                      |       0.0196 |        0.0285 |   1.0769 |   1.8630 |        -0.0083 |           166 |        3.6724 |            0.8400 |
| rank_pct      |               6 |     500.0000 |            3.0000 | rank_pct_target6m_minv500_lb126_entry0p1_backwardation_only_costx3p0        |       0.0033 |        0.0028 |   0.4625 |   0.7960 |        -0.0022 |             4 |        0.7635 |            0.3741 |
| rank_pct      |               1 |    1000.0000 |            1.0000 | rank_pct_target1m_minv1000_lb126_entry0p1_both_costx1p0                     |       0.0185 |        0.0153 |   1.0797 |   1.8678 |        -0.0081 |           116 |        4.8507 |            0.7679 |
| rank_pct      |               6 |    1000.0000 |            3.0000 | rank_pct_target6m_minv1000_lb252_entry0p05_contango_only_costx3p0           |       0.0000 |        0.0000 | nan      | nan      |         0.0000 |             0 |      nan      |            0.0000 |

## 1x Cost Tenor And Volume Robustness

| detector      |   target_months |   min_volume |   cost_multiplier | variant                                                                    |   net_return |   cost_return |   sharpe |    tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|:--------------|----------------:|-------------:|------------------:|:---------------------------------------------------------------------------|-------------:|--------------:|---------:|---------:|---------------:|--------------:|--------------:|------------------:|
| curve_md      |               1 |      10.0000 |            1.0000 | curve_md_target1m_minv10_lb126_md2p5_comp1p25_both_costx1p0                |       0.0562 |        0.0648 |   2.7144 |   4.6998 |        -0.0045 |           312 |        8.8684 |            0.6777 |
| curve_md      |               1 |     500.0000 |            1.0000 | curve_md_target1m_minv500_lb126_md2p5_comp1p25_both_costx1p0               |       0.0046 |        0.0157 |   0.2739 |   0.4738 |        -0.0094 |           118 |        1.0917 |            0.4789 |
| curve_md      |               1 |    1000.0000 |            1.0000 | curve_md_target1m_minv1000_lb126_md3p0_comp1p25_both_costx1p0              |       0.0036 |        0.0045 |   0.2445 |   0.4229 |        -0.0075 |            44 |        2.0351 |            0.2397 |
| curve_md      |               3 |      10.0000 |            1.0000 | curve_md_target3m_minv10_lb126_md2p5_comp1p25_both_costx1p0                |       0.0525 |        0.0464 |   2.1826 |   3.7790 |        -0.0090 |           239 |        6.1594 |            0.7011 |
| curve_md      |               3 |     500.0000 |            1.0000 | curve_md_target3m_minv500_lb252_md2p5_comp1p25_both_costx1p0               |       0.0147 |        0.0124 |   0.7464 |   1.2912 |        -0.0085 |            75 |        2.1505 |            0.5834 |
| curve_md      |               3 |    1000.0000 |            1.0000 | curve_md_target3m_minv1000_lb126_md3p0_comp1p25_both_costx1p0              |       0.0035 |        0.0038 |   0.2058 |   0.3560 |        -0.0082 |            32 |        0.6210 |            0.3469 |
| curve_md      |               6 |      10.0000 |            1.0000 | curve_md_target6m_minv10_lb126_md2p5_comp1p25_both_costx1p0                |       0.0316 |        0.0274 |   1.3644 |   2.3603 |        -0.0075 |           179 |        3.5737 |            0.6362 |
| curve_md      |               6 |     500.0000 |            1.0000 | curve_md_target6m_minv500_lb126_md3p0_comp1p25_both_costx1p0               |       0.0100 |        0.0018 |   1.1990 |   2.0636 |        -0.0012 |            19 |        1.5317 |            0.2867 |
| curve_md      |               6 |    1000.0000 |            1.0000 | curve_md_target6m_minv1000_lb126_md3p0_comp1p25_both_costx1p0              |      -0.0019 |        0.0006 |  -0.4640 |  -0.7953 |        -0.0036 |            10 |       -0.7274 |            0.3662 |
| mad_z         |               1 |      10.0000 |            1.0000 | mad_z_target1m_minv10_lb126_entry1p5_both_costx1p0                         |       0.0675 |        0.0812 |   2.9208 |   5.0573 |        -0.0053 |           321 |        7.8227 |            0.8544 |
| mad_z         |               1 |     500.0000 |            1.0000 | mad_z_target1m_minv500_lb126_entry1p5_both_costx1p0                        |       0.0214 |        0.0241 |   1.1822 |   2.0451 |        -0.0087 |           133 |        3.8387 |            0.8052 |
| mad_z         |               1 |    1000.0000 |            1.0000 | mad_z_target1m_minv1000_lb126_entry1p5_both_costx1p0                       |       0.0158 |        0.0131 |   0.9713 |   1.6802 |        -0.0081 |            90 |        4.1129 |            0.7346 |
| mad_z         |               3 |      10.0000 |            1.0000 | mad_z_target3m_minv10_lb126_entry1p5_both_costx1p0                         |       0.0659 |        0.0514 |   2.5285 |   4.3780 |        -0.0091 |           230 |        6.7250 |            0.8419 |
| mad_z         |               3 |     500.0000 |            1.0000 | mad_z_target3m_minv500_lb126_entry1p5_both_costx1p0                        |       0.0131 |        0.0141 |   0.6340 |   1.0967 |        -0.0096 |            69 |        1.6767 |            0.7306 |
| mad_z         |               3 |    1000.0000 |            1.0000 | mad_z_target3m_minv1000_lb126_entry2p0_contango_only_costx1p0              |       0.0045 |        0.0021 |   0.7358 |   1.2728 |        -0.0028 |            14 |        1.4620 |            0.2047 |
| mad_z         |               6 |      10.0000 |            1.0000 | mad_z_target6m_minv10_lb126_entry1p5_both_costx1p0                         |       0.0314 |        0.0267 |   1.3006 |   2.2499 |        -0.0072 |           164 |        3.0969 |            0.7659 |
| mad_z         |               6 |     500.0000 |            1.0000 | mad_z_target6m_minv500_lb252_entry2p0_backwardation_only_costx1p0          |       0.0046 |        0.0009 |   0.6528 |   1.1235 |        -0.0012 |             4 |        1.1857 |            0.3986 |
| mad_z         |               6 |    1000.0000 |            1.0000 | mad_z_target6m_minv1000_lb252_entry2p0_contango_only_costx1p0              |       0.0000 |        0.0000 | nan      | nan      |         0.0000 |             0 |      nan      |            0.0000 |
| quantile_band |               1 |      10.0000 |            1.0000 | quantile_band_target1m_minv10_lb126_entry0p1_both_costx1p0                 |       0.0661 |        0.0930 |   2.8562 |   4.9454 |        -0.0053 |           358 |        7.7238 |            0.8544 |
| quantile_band |               1 |     500.0000 |            1.0000 | quantile_band_target1m_minv500_lb126_entry0p1_both_costx1p0                |       0.0208 |        0.0293 |   1.1266 |   1.9488 |        -0.0081 |           170 |        3.7831 |            0.8424 |
| quantile_band |               1 |    1000.0000 |            1.0000 | quantile_band_target1m_minv1000_lb126_entry0p1_both_costx1p0               |       0.0184 |        0.0154 |   1.0737 |   1.8573 |        -0.0080 |           116 |        4.8005 |            0.7705 |
| quantile_band |               3 |      10.0000 |            1.0000 | quantile_band_target3m_minv10_lb126_entry0p1_both_costx1p0                 |       0.0721 |        0.0626 |   2.5829 |   4.4722 |        -0.0094 |           277 |        7.4005 |            0.8720 |
| quantile_band |               3 |     500.0000 |            1.0000 | quantile_band_target3m_minv500_lb126_entry0p1_both_costx1p0                |       0.0148 |        0.0165 |   0.7016 |   1.2136 |        -0.0092 |            88 |        1.8275 |            0.7884 |
| quantile_band |               3 |    1000.0000 |            1.0000 | quantile_band_target3m_minv1000_lb126_entry0p1_contango_only_costx1p0      |       0.0049 |        0.0044 |   0.6143 |   1.0626 |        -0.0040 |            29 |        1.4975 |            0.3734 |
| quantile_band |               6 |      10.0000 |            1.0000 | quantile_band_target6m_minv10_lb252_entry0p05_both_costx1p0                |       0.0310 |        0.0221 |   1.2888 |   2.2294 |        -0.0066 |           138 |        2.7920 |            0.7242 |
| quantile_band |               6 |     500.0000 |            1.0000 | quantile_band_target6m_minv500_lb126_entry0p1_backwardation_only_costx1p0  |       0.0051 |        0.0009 |   0.7160 |   1.2324 |        -0.0012 |             4 |        1.3595 |            0.3741 |
| quantile_band |               6 |    1000.0000 |            1.0000 | quantile_band_target6m_minv1000_lb252_entry0p1_backwardation_only_costx1p0 |       0.0000 |        0.0000 | nan      | nan      |         0.0000 |             0 |      nan      |            0.0000 |
| rank_pct      |               1 |      10.0000 |            1.0000 | rank_pct_target1m_minv10_lb126_entry0p1_both_costx1p0                      |       0.0646 |        0.0915 |   2.8271 |   4.8949 |        -0.0053 |           358 |        7.8880 |            0.8533 |
| rank_pct      |               1 |     500.0000 |            1.0000 | rank_pct_target1m_minv500_lb126_entry0p1_both_costx1p0                     |       0.0196 |        0.0285 |   1.0769 |   1.8630 |        -0.0083 |           166 |        3.6724 |            0.8400 |
| rank_pct      |               1 |    1000.0000 |            1.0000 | rank_pct_target1m_minv1000_lb126_entry0p1_both_costx1p0                    |       0.0185 |        0.0153 |   1.0797 |   1.8678 |        -0.0081 |           116 |        4.8507 |            0.7679 |
| rank_pct      |               3 |      10.0000 |            1.0000 | rank_pct_target3m_minv10_lb126_entry0p1_both_costx1p0                      |       0.0729 |        0.0625 |   2.6333 |   4.5594 |        -0.0094 |           275 |        7.4693 |            0.8699 |
| rank_pct      |               3 |     500.0000 |            1.0000 | rank_pct_target3m_minv500_lb126_entry0p1_both_costx1p0                     |       0.0150 |        0.0165 |   0.7097 |   1.2277 |        -0.0092 |            88 |        1.8467 |            0.7884 |
| rank_pct      |               3 |    1000.0000 |            1.0000 | rank_pct_target3m_minv1000_lb126_entry0p1_contango_only_costx1p0           |       0.0051 |        0.0044 |   0.6400 |   1.1071 |        -0.0040 |            29 |        1.5509 |            0.3719 |
| rank_pct      |               6 |      10.0000 |            1.0000 | rank_pct_target6m_minv10_lb252_entry0p05_both_costx1p0                     |       0.0298 |        0.0214 |   1.2469 |   2.1570 |        -0.0066 |           135 |        2.6985 |            0.7138 |
| rank_pct      |               6 |     500.0000 |            1.0000 | rank_pct_target6m_minv500_lb252_entry0p1_backwardation_only_costx1p0       |       0.0051 |        0.0009 |   0.7160 |   1.2324 |        -0.0012 |             4 |        1.3595 |            0.3741 |
| rank_pct      |               6 |    1000.0000 |            1.0000 | rank_pct_target6m_minv1000_lb252_entry0p1_backwardation_only_costx1p0      |       0.0000 |        0.0000 | nan      | nan      |         0.0000 |             0 |      nan      |            0.0000 |

## Split Metrics

| split   |   gross_return |   cost_return |   net_return |   sharpe |   tstat |   max_drawdown |   events |   bars |
|:--------|---------------:|--------------:|-------------:|---------:|--------:|---------------:|---------:|-------:|
| full    |         0.1302 |        0.0625 |       0.0729 |   2.6333 |  4.5594 |        -0.0094 |      275 |    930 |
| 2023    |         0.0099 |        0.0062 |       0.0043 |   2.4626 |  1.7761 |        -0.0008 |       35 |    163 |
| 2024    |         0.0337 |        0.0214 |       0.0147 |   1.3984 |  1.3960 |        -0.0094 |       83 |    310 |
| 2025    |         0.0505 |        0.0237 |       0.0286 |   2.9727 |  2.9636 |        -0.0037 |       99 |    311 |
| 2026    |         0.0361 |        0.0111 |       0.0253 |   5.5454 |  3.7944 |        -0.0017 |       58 |    146 |

## Event Summary For Best Variant

| root   | side               |   events |   mean_weighted_net_return |   mean_gross_spread_return |   win_rate |   event_tstat |   mean_duration_days |   mean_rolls |
|:-------|:-------------------|---------:|---------------------------:|---------------------------:|-----------:|--------------:|---------------------:|-------------:|
| GC     | fade_backwardation |       36 |                     0.0001 |                     0.0010 |     0.6667 |        3.7126 |               8.6111 |       1.0000 |
| GC     | fade_contango      |       21 |                     0.0002 |                     0.0012 |     0.9048 |        3.2384 |               8.3810 |       0.6190 |
| HG     | fade_backwardation |       20 |                     0.0004 |                     0.0026 |     0.9000 |        1.0861 |               6.3500 |       0.7000 |
| HG     | fade_contango      |       33 |                     0.0003 |                     0.0023 |     0.6970 |        3.3817 |               9.9697 |       1.4848 |
| PA     | fade_backwardation |       15 |                     0.0005 |                     0.0038 |     0.8667 |        3.0705 |               3.5333 |       0.0667 |
| PA     | fade_contango      |       38 |                     0.0005 |                     0.0036 |     0.8158 |        4.4292 |               5.9474 |       0.0000 |
| PL     | fade_backwardation |       16 |                     0.0004 |                     0.0029 |     0.8750 |        4.6479 |              12.3750 |       0.1875 |
| PL     | fade_contango      |       23 |                     0.0000 |                     0.0008 |     0.6087 |        0.2726 |              13.6522 |       0.0870 |
| SI     | fade_backwardation |       43 |                     0.0003 |                     0.0020 |     0.8140 |        3.7962 |               6.5116 |       0.1860 |
| SI     | fade_contango      |       30 |                     0.0004 |                     0.0024 |     0.8333 |        3.4246 |               4.5000 |       0.1667 |

## Interpretation

The best robust-detector variant is `rank_pct_target3m_minv10_lb126_entry0p1_both_costx1p0` with net cumulative log return `0.0729`, t-stat `4.56`, and event t-stat `7.47`.

This experiment asks whether the curve-basis edge survives non-Gaussian extreme definitions. A detector only advances if it remains positive under synchronized marks, liquid volume filters, and conservative cost multipliers.

## Files

- `strategy_metrics.csv`
- `detector_best.csv`
- `volume_cost_robustness.csv`
- `target_volume_robustness_1x.csv`
- `best_strategy_returns.csv`
- `event_log.csv`
- `split_metrics.csv`
- `root_event_summary.csv`
- `best_strategy_equity.png`
- `detector_best.png`
- `top_variant_metrics.png`