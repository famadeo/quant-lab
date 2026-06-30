# HYP-0031 Metals Convenience-Yield Basis Backtest With Synchronized 5m Marks

## Design

- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.
- Pair selection is still daily: liquid front versus first deferred contract at the target tenor.
- Entry and next-day marks use the last exact shared 5-minute timestamp for the selected near/far contracts.
- Signal, event exits, costs, position sizing, and robustness grid match HYP-0030.
- This is stricter than daily last-trade closes and removes asynchronous near/far marks.

## Best Variant

| variant                                               |   min_volume |   cost_multiplier |   net_return |   cost_return |   sharpe |   tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|:------------------------------------------------------|-------------:|------------------:|-------------:|--------------:|---------:|--------:|---------------:|--------------:|--------------:|------------------:|
| target3m_minv10_lb126_entry1p5_exit0p25_both_costx1p0 |      10.0000 |            1.0000 |       0.0654 |        0.0454 |   2.6476 |  4.5843 |        -0.0091 |           196 |        6.6727 |            0.8409 |

## Volume And Cost Robustness

|   target_months |   min_volume |   cost_multiplier | variant                                                              |   net_return |   cost_return |   sharpe |   tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|----------------:|-------------:|------------------:|:---------------------------------------------------------------------|-------------:|--------------:|---------:|--------:|---------------:|--------------:|--------------:|------------------:|
|               3 |      10.0000 |            1.0000 | target3m_minv10_lb126_entry1p5_exit0p25_both_costx1p0                |       0.0654 |        0.0454 |   2.6476 |  4.5843 |        -0.0091 |           196 |        6.6727 |            0.8409 |
|               3 |      10.0000 |            3.0000 | target3m_minv10_lb126_entry2p0_exit0p25_both_costx3p0                |       0.0084 |        0.0735 |   0.3887 |  0.6730 |        -0.0142 |           120 |        2.9881 |            0.6828 |
|               1 |     500.0000 |            1.0000 | target1m_minv500_lb126_entry1p5_exit0p25_both_costx1p0               |       0.0168 |        0.0197 |   0.9682 |  1.6749 |        -0.0081 |           108 |        2.9748 |            0.7593 |
|               6 |     500.0000 |            3.0000 | target6m_minv500_lb252_entry1p5_exit0p25_backwardation_only_costx3p0 |       0.0039 |        0.0026 |   0.5736 |  0.9873 |        -0.0021 |             4 |        0.7466 |            0.3671 |
|               1 |    1000.0000 |            1.0000 | target1m_minv1000_lb126_entry1p5_exit0p25_both_costx1p0              |       0.0134 |        0.0111 |   0.8441 |  1.4602 |        -0.0082 |            82 |        5.3268 |            0.6333 |
|               3 |    1000.0000 |            3.0000 | target3m_minv1000_lb126_entry2p0_exit0p25_contango_only_costx3p0     |       0.0013 |        0.0048 |   0.2622 |  0.4537 |        -0.0017 |             9 |        1.1604 |            0.1484 |

## 1x Cost Tenor And Volume Robustness

|   target_months |   min_volume |   cost_multiplier | variant                                                               |   net_return |   cost_return |   sharpe |    tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|----------------:|-------------:|------------------:|:----------------------------------------------------------------------|-------------:|--------------:|---------:|---------:|---------------:|--------------:|--------------:|------------------:|
|               1 |      10.0000 |            1.0000 | target1m_minv10_lb126_entry1p5_exit0p25_both_costx1p0                 |       0.0600 |        0.0549 |   2.9835 |   5.1659 |        -0.0052 |           239 |        8.8573 |            0.7516 |
|               1 |     500.0000 |            1.0000 | target1m_minv500_lb126_entry1p5_exit0p25_both_costx1p0                |       0.0168 |        0.0197 |   0.9682 |   1.6749 |        -0.0081 |           108 |        2.9748 |            0.7593 |
|               1 |    1000.0000 |            1.0000 | target1m_minv1000_lb126_entry1p5_exit0p25_both_costx1p0               |       0.0134 |        0.0111 |   0.8441 |   1.4602 |        -0.0082 |            82 |        5.3268 |            0.6333 |
|               3 |      10.0000 |            1.0000 | target3m_minv10_lb126_entry1p5_exit0p25_both_costx1p0                 |       0.0654 |        0.0454 |   2.6476 |   4.5843 |        -0.0091 |           196 |        6.6727 |            0.8409 |
|               3 |     500.0000 |            1.0000 | target3m_minv500_lb126_entry2p0_exit0p25_both_costx1p0                |       0.0155 |        0.0092 |   0.7948 |   1.3748 |        -0.0095 |            47 |        2.4163 |            0.5821 |
|               3 |    1000.0000 |            1.0000 | target3m_minv1000_lb126_entry2p0_exit0p25_contango_only_costx1p0      |       0.0045 |        0.0016 |   0.8784 |   1.5195 |        -0.0011 |             9 |        2.0611 |            0.1484 |
|               6 |      10.0000 |            1.0000 | target6m_minv10_lb252_entry1p5_exit0p25_both_costx1p0                 |       0.0287 |        0.0233 |   1.2241 |   2.1176 |        -0.0069 |           139 |        2.6906 |            0.7335 |
|               6 |     500.0000 |            1.0000 | target6m_minv500_lb252_entry1p5_exit0p25_backwardation_only_costx1p0  |       0.0056 |        0.0009 |   0.8324 |   1.4327 |        -0.0011 |             4 |        1.1792 |            0.3671 |
|               6 |    1000.0000 |            1.0000 | target6m_minv1000_lb252_entry1p5_exit0p25_backwardation_only_costx1p0 |       0.0000 |        0.0000 | nan      | nan      |         0.0000 |             0 |      nan      |            0.0000 |

## Split Metrics

| split   |   gross_return |   cost_return |   net_return |   sharpe |   tstat |   max_drawdown |   events |   bars |
|:--------|---------------:|--------------:|-------------:|---------:|--------:|---------------:|---------:|-------:|
| full    |         0.1064 |        0.0454 |       0.0654 |   2.6476 |  4.5843 |        -0.0091 |      196 |    930 |
| 2023    |         0.0083 |        0.0043 |       0.0044 |   2.4477 |  1.7654 |        -0.0012 |       27 |    163 |
| 2024    |         0.0249 |        0.0162 |       0.0110 |   1.0969 |  1.0950 |        -0.0091 |       59 |    310 |
| 2025    |         0.0433 |        0.0163 |       0.0285 |   3.6860 |  3.6747 |        -0.0036 |       68 |    311 |
| 2026    |         0.0299 |        0.0087 |       0.0215 |   5.2660 |  3.6032 |        -0.0014 |       42 |    146 |

## Event Summary For Best Variant

| root   | side               |   events |   mean_weighted_net_return |   mean_gross_spread_return |   win_rate |   event_tstat |   mean_duration_days |   mean_rolls |
|:-------|:-------------------|---------:|---------------------------:|---------------------------:|-----------:|--------------:|---------------------:|-------------:|
| GC     | fade_backwardation |       28 |                     0.0001 |                     0.0010 |     0.6429 |        3.0712 |               9.8214 |       1.1071 |
| GC     | fade_contango      |       16 |                     0.0002 |                     0.0014 |     0.8750 |        3.1638 |               8.5625 |       0.6250 |
| HG     | fade_backwardation |       13 |                     0.0005 |                     0.0029 |     0.8462 |        0.7953 |               7.7692 |       0.9231 |
| HG     | fade_contango      |       21 |                     0.0004 |                     0.0029 |     0.7143 |        3.2152 |               9.0476 |       1.6667 |
| PA     | fade_backwardation |       10 |                     0.0007 |                     0.0047 |     0.8000 |        3.4582 |               3.9000 |       0.1000 |
| PA     | fade_contango      |       31 |                     0.0005 |                     0.0039 |     0.8065 |        4.0176 |               6.3226 |       0.0000 |
| PL     | fade_backwardation |       12 |                     0.0005 |                     0.0034 |     0.8333 |        4.4033 |              16.0000 |       0.2500 |
| PL     | fade_contango      |       15 |                     0.0000 |                     0.0008 |     0.5333 |        0.2026 |              16.2667 |       0.1333 |
| SI     | fade_backwardation |       27 |                     0.0004 |                     0.0028 |     0.8148 |        3.4288 |               8.0741 |       0.2963 |
| SI     | fade_contango      |       23 |                     0.0004 |                     0.0023 |     0.8261 |        3.9977 |               4.5652 |       0.1304 |

## Sync Mark Coverage

| root   |   min_volume |   target_months |   rows |   entry_marked |   return_marked |   median_common_5m_obs |
|:-------|-------------:|----------------:|-------:|---------------:|----------------:|-----------------------:|
| GC     |        10.00 |               1 |    927 |            927 |             913 |                  93.00 |
| GC     |        10.00 |               3 |    884 |            884 |             850 |                  59.00 |
| GC     |        10.00 |               6 |    837 |            837 |             788 |                  46.00 |
| GC     |       500.00 |               1 |    746 |            746 |             746 |                 197.00 |
| GC     |       500.00 |               3 |    420 |            420 |             418 |                 143.50 |
| GC     |       500.00 |               6 |    170 |            170 |             167 |                 112.50 |
| GC     |      1000.00 |               1 |    670 |            670 |             670 |                 219.00 |
| GC     |      1000.00 |               3 |    192 |            192 |             192 |                 186.50 |
| GC     |      1000.00 |               6 |     37 |             37 |              35 |                 123.00 |
| HG     |        10.00 |               1 |    903 |            903 |             824 |                  22.00 |
| HG     |        10.00 |               3 |    865 |            865 |             815 |                  87.00 |
| HG     |        10.00 |               6 |    780 |            780 |             686 |                  41.00 |
| HG     |       500.00 |               1 |    765 |            765 |             763 |                 207.00 |
| HG     |       500.00 |               3 |    622 |            622 |             608 |                 160.00 |
| HG     |       500.00 |               6 |    132 |            132 |             123 |                  95.00 |
| HG     |      1000.00 |               1 |    693 |            693 |             691 |                 216.00 |
| HG     |      1000.00 |               3 |    439 |            439 |             432 |                 186.00 |
| HG     |      1000.00 |               6 |     34 |             34 |              31 |                 115.00 |
| PA     |        10.00 |               1 |    730 |            730 |             669 |                  30.00 |
| PA     |        10.00 |               3 |    730 |            730 |             675 |                  34.00 |
| PA     |        10.00 |               6 |     64 |             64 |              42 |                   6.00 |
| PA     |       500.00 |               1 |     76 |             76 |              76 |                 136.50 |
| PA     |       500.00 |               3 |     76 |             76 |              76 |                 136.50 |
| PA     |      1000.00 |               1 |     34 |             34 |              34 |                 169.50 |
| PA     |      1000.00 |               3 |     34 |             34 |              34 |                 169.50 |
| PL     |        10.00 |               1 |    894 |            894 |             872 |                 115.00 |
| PL     |        10.00 |               3 |    894 |            894 |             885 |                 141.00 |
| PL     |        10.00 |               6 |    748 |            748 |             691 |                  44.00 |
| PL     |       500.00 |               1 |    554 |            554 |             550 |                 168.50 |
| PL     |       500.00 |               3 |    554 |            554 |             550 |                 169.00 |
| PL     |       500.00 |               6 |     27 |             27 |              25 |                 102.00 |
| PL     |      1000.00 |               1 |    309 |            309 |             309 |                 203.00 |
| PL     |      1000.00 |               3 |    309 |            309 |             309 |                 203.00 |
| PL     |      1000.00 |               6 |      3 |              3 |               3 |                 163.00 |
| SI     |        10.00 |               1 |    915 |            915 |             893 |                  50.00 |
| SI     |        10.00 |               3 |    871 |            871 |             835 |                  67.00 |
| SI     |        10.00 |               6 |    668 |            668 |             585 |                  15.00 |
| SI     |       500.00 |               1 |    682 |            682 |             681 |                 176.00 |
| SI     |       500.00 |               3 |    339 |            339 |             337 |                 155.00 |
| SI     |       500.00 |               6 |      8 |              8 |               8 |                  78.50 |
| SI     |      1000.00 |               1 |    466 |            466 |             466 |                 212.00 |
| SI     |      1000.00 |               3 |    184 |            184 |             184 |                 196.00 |

## Interpretation

The best synchronized-mark variant is `target3m_minv10_lb126_entry1p5_exit0p25_both_costx1p0` with net cumulative log return `0.0654`, t-stat `4.58`, and event t-stat `6.67`.

If this result is materially weaker than HYP-0030, the daily-close result was likely benefiting from asynchronous deferred-contract marks. If it survives, the curve-basis signal has passed a more realistic pricing gate.

## Files

- `curve_panel.parquet`
- `sync_mark_coverage.csv`
- `strategy_metrics.csv`
- `best_strategy_returns.csv`
- `event_log.csv`
- `split_metrics.csv`
- `root_event_summary.csv`
- `volume_cost_robustness.csv`
- `target_volume_robustness_1x.csv`
- `best_strategy_equity.png`
- `root_event_summary.png`
- `top_variant_metrics.png`