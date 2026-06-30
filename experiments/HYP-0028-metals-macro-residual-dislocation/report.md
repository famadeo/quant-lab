# HYP-0028 Metals Macro-Residual Dislocation Strategy

Completed at `2026-06-26T12:38:10.118789+00:00`.

## Design

- Frequency: daily continuous futures, because aligned 5m macro factors only cover about one month.
- Residual model: rolling lagged OLS per metal on USD, rates-price, CL, and leave-one-out metals complex.
- Dislocation state: residual z-scores, rolling residual-correlation MST neighbors, and residual-cloud Mahalanobis distance.
- Entry: root-vs-MST-neighbor residual spread exceeds z threshold and residual-cloud MD exceeds rolling threshold.
- Exit: spread normalizes, MD returns to normal, sign flips, or optionally MST topology changes. No fixed-time exit.
- Execution: signal at close `t`; position earns return at `t+1`; metal costs charged on turnover.
- PnL focus: `net_residual_return` is the macro-hedged residual alpha test. `net_raw_return` is the unhedged metal-basket implementation check.

## Coverage

- Return/factor span: `2010-06-07` to `2024-11-29`.
- Factor panel span: `2010-06-07` to `2024-11-29`.
- Explicit carry-curve adjustment is not included in this daily prototype; continuous roll-adjusted futures returns are used.
- Macro hedge costs are not included, so residual PnL is optimistic relative to a fully executable hedge package.

## Best Variant

| variant                              |   net_residual_return |   cost_return |   net_raw_return |   residual_sharpe |   residual_tstat |   max_drawdown |   event_count |   mean_event_return |   event_tstat |
|:-------------------------------------|----------------------:|--------------:|-----------------:|------------------:|-----------------:|---------------:|--------------:|--------------------:|--------------:|
| z2.5_exit0.5_mdq90_normq50_topo_hold |                0.1281 |        0.1454 |           0.1237 |            0.1635 |           0.6547 |        -0.1769 |           592 |              0.0022 |        2.0079 |

## Best Variant Splits

| split         | start                     | end                       |   net_residual_return |   net_raw_return |   cost_return |   residual_sharpe |   residual_tstat |   events |
|:--------------|:--------------------------|:--------------------------|----------------------:|-----------------:|--------------:|------------------:|-----------------:|---------:|
| full          | 2011-11-07 00:00:00+00:00 | 2024-11-29 00:00:00+00:00 |                0.1281 |           0.1237 |        0.1454 |            0.1635 |           0.6547 |      592 |
| pre_2020      | 2011-11-07 00:00:00+00:00 | 2019-12-31 00:00:00+00:00 |               -0.0729 |          -0.0279 |        0.0888 |           -0.2099 |          -0.6633 |      329 |
| post_2020     | 2020-01-01 00:00:00+00:00 | 2024-11-29 00:00:00+00:00 |                0.2010 |           0.1516 |        0.0566 |            0.5049 |           1.2421 |      263 |
| trade_overlap | 2023-06-22 00:00:00+00:00 | 2024-11-29 00:00:00+00:00 |                0.0122 |           0.0055 |        0.0168 |            0.1239 |           0.1654 |       68 |

## Top Metrics

| variant                               |   net_residual_return |   cost_return |   net_raw_return |   residual_sharpe |   residual_tstat |   event_count |   event_tstat |   mean_duration_days |
|:--------------------------------------|----------------------:|--------------:|-----------------:|------------------:|-----------------:|--------------:|--------------:|---------------------:|
| z2.5_exit0.5_mdq90_normq50_topo_hold  |                0.1281 |        0.1454 |           0.1237 |            0.1635 |           0.6547 |           592 |        2.0079 |               1.6689 |
| z2.5_exit0.5_mdq95_normq50_topo_hold  |                0.1238 |        0.0812 |           0.1144 |            0.1985 |           0.7950 |           368 |        1.7143 |               1.6929 |
| z2.5_exit0.5_mdq95_normq50_topo_exit  |                0.1139 |        0.0809 |           0.0910 |            0.1833 |           0.7342 |           370 |        1.7052 |               1.6405 |
| z2.5_exit0.5_mdq90_normq50_topo_exit  |                0.1129 |        0.1454 |           0.0988 |            0.1444 |           0.5782 |           596 |        1.9778 |               1.6309 |
| z2.5_exit0.25_mdq95_normq50_topo_hold |                0.1080 |        0.0808 |           0.0888 |            0.1731 |           0.6933 |           368 |        1.7228 |               1.7391 |
| z2.5_exit0.25_mdq90_normq50_topo_hold |                0.1075 |        0.1445 |           0.0660 |            0.1386 |           0.5551 |           592 |        2.0321 |               1.7280 |
| z2.5_exit0.25_mdq95_normq50_topo_exit |                0.0930 |        0.0807 |           0.0679 |            0.1500 |           0.6006 |           370 |        1.6714 |               1.6838 |
| z2.5_exit0.25_mdq90_normq50_topo_exit |                0.0872 |        0.1446 |           0.0436 |            0.1129 |           0.4520 |           596 |        1.9692 |               1.6879 |
| z1.5_exit0.5_mdq90_normq50_topo_exit  |                0.0872 |        0.1785 |           0.1092 |            0.1203 |           0.4819 |          1046 |        2.0153 |               1.5966 |
| z1.5_exit0.5_mdq90_normq50_topo_hold  |                0.0862 |        0.1804 |           0.1156 |            0.1183 |           0.4736 |          1038 |        1.9886 |               1.6358 |
| z1.5_exit0.5_mdq95_normq50_topo_hold  |                0.0763 |        0.0979 |           0.0653 |            0.1274 |           0.5104 |           588 |        1.2340 |               1.6582 |
| z2.5_exit0.25_mdq95_normq75_topo_hold |                0.0732 |        0.0805 |           0.0961 |            0.1218 |           0.4879 |           368 |        1.4516 |               1.5815 |

## Event Direction Notes

| variant                              | root   |    sign |   events |   mean_event_return |   median_event_return |   event_tstat |   win_rate |   mean_duration_days |
|:-------------------------------------|:-------|--------:|---------:|--------------------:|----------------------:|--------------:|-----------:|---------------------:|
| z2.5_exit0.5_mdq90_normq50_topo_hold | SI     | -1.0000 |       49 |              0.0049 |                0.0038 |        2.7058 |     0.7347 |               1.4694 |
| z2.5_exit0.5_mdq90_normq50_topo_hold | PL     | -1.0000 |       51 |              0.0041 |                0.0042 |        1.2816 |     0.6275 |               1.7647 |
| z2.5_exit0.5_mdq90_normq50_topo_hold | GC     |  1.0000 |       40 |              0.0037 |                0.0010 |        0.9959 |     0.5500 |               1.7250 |
| z2.5_exit0.5_mdq90_normq50_topo_hold | HG     | -1.0000 |       87 |              0.0036 |                0.0014 |        1.4875 |     0.5862 |               1.5632 |
| z2.5_exit0.5_mdq90_normq50_topo_hold | SI     |  1.0000 |       42 |              0.0029 |                0.0020 |        0.9827 |     0.6190 |               1.6905 |
| z2.5_exit0.5_mdq90_normq50_topo_hold | PA     |  1.0000 |       61 |              0.0022 |                0.0011 |        0.5641 |     0.5738 |               1.8197 |
| z2.5_exit0.5_mdq90_normq50_topo_hold | PA     | -1.0000 |       77 |              0.0013 |                0.0015 |        0.2386 |     0.5325 |               1.6364 |
| z2.5_exit0.5_mdq90_normq50_topo_hold | PL     |  1.0000 |       68 |              0.0011 |                0.0005 |        0.4795 |     0.5147 |               1.6176 |
| z2.5_exit0.5_mdq90_normq50_topo_hold | HG     |  1.0000 |       81 |              0.0010 |                0.0009 |        0.3952 |     0.5309 |               1.6543 |
| z2.5_exit0.5_mdq90_normq50_topo_hold | GC     | -1.0000 |       36 |             -0.0030 |               -0.0002 |       -0.9456 |     0.5000 |               1.9167 |

## Files

- `strategy_metrics.csv`
- `split_metrics.csv`
- `event_log.csv`
- `event_summary.csv`
- `macro_residual_returns.parquet`
- `rolling_macro_betas.parquet`
- `residual_dislocation_state.parquet`
- `best_strategy_returns.csv`
- `best_strategy_equity.png`
- `top_variant_metrics.png`
