# HYP-0030 Metals Convenience-Yield Basis Backtest

## Design

- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.
- Data: raw 1-minute outright futures, collapsed to daily contract closes.
- Pair selection: liquid front contract versus first deferred contract at least the target tenor away.
- Robustness controls: minimum leg volume variants, cost multipliers, and a `120` minute max near/far last-trade timestamp gap.
- Signal: annualized `log(far/front)` carry z-score using lagged rolling statistics.
- Low carry / backwardation shock: long deferred, short front, expecting carry normalization.
- High carry / contango shock: short deferred, long front, expecting carry normalization.
- Exit: event-based normalization of the carry z-score; no fixed holding time.
- Costs: two-leg futures execution cost using prior per-side MBP1 estimates.

## Coverage

- Start: `2023-06-22`.
- End: `2026-06-21`.

## Best Variant

| variant                                               |   min_volume |   cost_multiplier |   net_return |   cost_return |   sharpe |   tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|:------------------------------------------------------|-------------:|------------------:|-------------:|--------------:|---------:|--------:|---------------:|--------------:|--------------:|------------------:|
| target3m_minv10_lb252_entry1p5_exit0p25_both_costx1p0 |      10.0000 |            1.0000 |       0.0813 |        0.0255 |   2.7308 |  4.7283 |        -0.0065 |           127 |        7.7724 |            0.4860 |

## Volume And Cost Robustness

|   target_months |   min_volume |   cost_multiplier | variant                                                 |   net_return |   cost_return |   sharpe |   tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|----------------:|-------------:|------------------:|:--------------------------------------------------------|-------------:|--------------:|---------:|--------:|---------------:|--------------:|--------------:|------------------:|
|               3 |      10.0000 |            1.0000 | target3m_minv10_lb252_entry1p5_exit0p25_both_costx1p0   |       0.0813 |        0.0255 |   2.7308 |  4.7283 |        -0.0065 |           127 |        7.7724 |            0.4860 |
|               1 |      10.0000 |            3.0000 | target1m_minv10_lb252_entry1p5_exit0p25_both_costx3p0   |       0.0512 |        0.0434 |   2.0768 |  3.5958 |        -0.0057 |            98 |        6.2119 |            0.2805 |
|               1 |     500.0000 |            1.0000 | target1m_minv500_lb126_entry1p5_exit0p25_both_costx1p0  |       0.0466 |        0.0203 |   2.2141 |  3.8302 |        -0.0071 |           134 |        5.9607 |            0.6241 |
|               3 |     500.0000 |            3.0000 | target3m_minv500_lb126_entry2p0_exit0p25_both_costx3p0  |       0.0172 |        0.0279 |   0.6962 |  1.2043 |        -0.0086 |            60 |        2.1642 |            0.3903 |
|               1 |    1000.0000 |            1.0000 | target1m_minv1000_lb126_entry1p5_exit0p25_both_costx1p0 |       0.0343 |        0.0119 |   1.8999 |  3.2866 |        -0.0078 |            95 |        8.1700 |            0.5641 |
|               1 |    1000.0000 |            3.0000 | target1m_minv1000_lb126_entry1p5_exit0p25_both_costx3p0 |       0.0111 |        0.0357 |   0.6246 |  1.0805 |        -0.0103 |            95 |        5.5148 |            0.5641 |

## 1x Cost Tenor And Volume Robustness

|   target_months |   min_volume |   cost_multiplier | variant                                                          |   net_return |   cost_return |   sharpe |    tstat |   max_drawdown |   event_count |   event_tstat |   active_fraction |
|----------------:|-------------:|------------------:|:-----------------------------------------------------------------|-------------:|--------------:|---------:|---------:|---------------:|--------------:|--------------:|------------------:|
|               1 |      10.0000 |            1.0000 | target1m_minv10_lb252_entry1p5_exit0p25_both_costx1p0            |       0.0751 |        0.0145 |   2.8492 |   4.9332 |        -0.0049 |            98 |        7.4150 |            0.2805 |
|               1 |     500.0000 |            1.0000 | target1m_minv500_lb126_entry1p5_exit0p25_both_costx1p0           |       0.0466 |        0.0203 |   2.2141 |   3.8302 |        -0.0071 |           134 |        5.9607 |            0.6241 |
|               1 |    1000.0000 |            1.0000 | target1m_minv1000_lb126_entry1p5_exit0p25_both_costx1p0          |       0.0343 |        0.0119 |   1.8999 |   3.2866 |        -0.0078 |            95 |        8.1700 |            0.5641 |
|               3 |      10.0000 |            1.0000 | target3m_minv10_lb252_entry1p5_exit0p25_both_costx1p0            |       0.0813 |        0.0255 |   2.7308 |   4.7283 |        -0.0065 |           127 |        7.7724 |            0.4860 |
|               3 |     500.0000 |            1.0000 | target3m_minv500_lb126_entry1p5_exit0p25_both_costx1p0           |       0.0384 |        0.0144 |   1.4990 |   2.5931 |        -0.0081 |            93 |        3.7783 |            0.5493 |
|               3 |    1000.0000 |            1.0000 | target3m_minv1000_lb126_entry1p5_exit0p25_both_costx1p0          |       0.0176 |        0.0062 |   0.8524 |   1.4745 |        -0.0081 |            42 |        2.5175 |            0.4094 |
|               6 |      10.0000 |            1.0000 | target6m_minv10_lb126_entry1p5_exit0p25_both_costx1p0            |       0.0504 |        0.0084 |   2.1564 |   3.7303 |        -0.0036 |            68 |        5.2596 |            0.1472 |
|               6 |     500.0000 |            1.0000 | target6m_minv500_lb126_entry1p5_exit0p25_both_costx1p0           |       0.0022 |        0.0006 |   0.3857 |   0.6638 |        -0.0031 |             8 |        1.4792 |            0.2552 |
|               6 |    1000.0000 |            1.0000 | target6m_minv1000_lb252_entry1p5_exit0p25_contango_only_costx1p0 |       0.0000 |        0.0000 | nan      | nan      |         0.0000 |             0 |      nan      |            0.0000 |

## Split Metrics

| split   |   gross_return |   cost_return |   net_return |   sharpe |   tstat |   max_drawdown |   events |   bars |
|:--------|---------------:|--------------:|-------------:|---------:|--------:|---------------:|---------:|-------:|
| full    |         0.1006 |        0.0255 |       0.0813 |   2.7308 |  4.7283 |        -0.0065 |      127 |    930 |
| 2023    |         0.0059 |        0.0024 |       0.0037 |   2.4239 |  1.7482 |        -0.0010 |       17 |    163 |
| 2024    |         0.0260 |        0.0082 |       0.0199 |   1.8280 |  1.8248 |        -0.0065 |       55 |    310 |
| 2025    |         0.0352 |        0.0088 |       0.0286 |   3.1970 |  3.1871 |        -0.0015 |       36 |    311 |
| 2026    |         0.0336 |        0.0060 |       0.0291 |   4.4776 |  3.0637 |        -0.0024 |       19 |    146 |

## Event Summary For Best Variant

| root   | side               |   events |   mean_weighted_net_return |   mean_gross_spread_return |   win_rate |   event_tstat |   mean_duration_days |   mean_rolls |
|:-------|:-------------------|---------:|---------------------------:|---------------------------:|-----------:|--------------:|---------------------:|-------------:|
| GC     | fade_backwardation |       25 |                     0.0003 |                     0.0015 |     0.6400 |        3.0424 |               2.7200 |       0.1200 |
| GC     | fade_contango      |        2 |                     0.0011 |                     0.0057 |     1.0000 |        4.8595 |               4.5000 |       0.0000 |
| HG     | fade_backwardation |        8 |                     0.0003 |                     0.0015 |     0.6250 |        0.4952 |               3.1250 |       0.0000 |
| HG     | fade_contango      |        6 |                     0.0007 |                     0.0035 |     0.8333 |        2.9200 |              10.6667 |       0.0000 |
| PA     | fade_backwardation |        4 |                     0.0026 |                     0.0142 |     1.0000 |        4.2790 |               1.7500 |       0.0000 |
| PA     | fade_contango      |       20 |                     0.0014 |                     0.0079 |     0.9000 |        5.3675 |               2.5500 |       0.0000 |
| PL     | fade_backwardation |       11 |                     0.0005 |                     0.0030 |     0.8182 |        3.0393 |              10.7273 |       0.0909 |
| PL     | fade_contango      |       33 |                     0.0003 |                     0.0023 |     0.7879 |        3.9247 |               6.5758 |       0.0606 |
| SI     | fade_backwardation |       12 |                     0.0007 |                     0.0037 |     0.9167 |        4.0732 |               2.6667 |       0.0000 |
| SI     | fade_contango      |        6 |                     0.0016 |                     0.0082 |     1.0000 |        1.9921 |               5.0000 |       0.0000 |

## Interpretation

The best variant is `target3m_minv10_lb252_entry1p5_exit0p25_both_costx1p0` with net cumulative log return `0.0813`, t-stat `4.73`, and event t-stat `7.77`.

This is a futures-only test of curve-basis mean reversion. It does not prove a pure physical arbitrage because spot storage, delivery optionality, warehouse location, and financing are not directly traded here.

## Files

- `curve_panel.parquet`
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