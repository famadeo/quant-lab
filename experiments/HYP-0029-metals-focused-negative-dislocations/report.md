# HYP-0029 Focused Negative Metals Residual Dislocations

Completed at `2026-06-26T12:47:30.344116+00:00`.

## Design

- Targets only `SI`, `PL`, and `HG` negative residual dislocations.
- Entry requires root-vs-MST-neighbor residual z below the threshold, the root residual z itself below the root threshold, and residual-cloud MD above its rolling threshold.
- Position is long the dislocated root and short its rolling MST neighbors.
- Exit is event-based: spread normalization, MD normalization, sign reversal, or optional topology change. No fixed holding time.
- Residual PnL includes a macro hedge cost haircut for USD, rates, CL, and the leave-one-out metals complex hedge.
- Carry proxy is available only from 2023-06-22 onward and is reported as an overlap diagnostic, not a full-history adjustment.

## Coverage

- Return/factor span: `2010-06-07` to `2024-11-29`.
- Carry proxy span: `2023-06-22` to `2026-06-21`.

## Best Variant

| variant                                             |   net_return_after_all_costs |   total_cost_return |   macro_hedge_cost_return |   trade_overlap_net |   trade_overlap_carry_adjusted_net |   sharpe_after_all_costs |   tstat_after_all_costs |   max_drawdown |   event_count |   event_tstat |
|:----------------------------------------------------|-----------------------------:|--------------------:|--------------------------:|--------------------:|-----------------------------------:|-------------------------:|------------------------:|---------------:|--------------:|--------------:|
| neg_z2_root1.5_exit0.25_mdq90_normq50_all_topo_hold |                       0.3526 |              0.0817 |                    0.0242 |              0.0663 |                             0.0663 |                   0.6887 |                  2.7578 |        -0.0390 |           148 |        3.6103 |

## Best Variant Splits

| split         |   net_return_after_all_costs |   carry_adjusted_net |   total_cost_return |   sharpe_after_all_costs |   tstat_after_all_costs |   events |
|:--------------|-----------------------------:|---------------------:|--------------------:|-------------------------:|------------------------:|---------:|
| full          |                       0.3526 |               0.3526 |              0.0817 |                   0.6887 |                  2.7578 |      148 |
| pre_2020      |                       0.0464 |               0.0464 |              0.0516 |                   0.1913 |                  0.6046 |       92 |
| post_2020     |                       0.3062 |               0.3062 |              0.0301 |                   1.2189 |                  2.9985 |       56 |
| trade_overlap |                       0.0663 |               0.0663 |              0.0077 |                   1.1020 |                  1.4710 |       15 |

## Top Variants

| variant                                                  |   net_return_after_all_costs |   total_cost_return |   trade_overlap_net |   trade_overlap_carry_adjusted_net |   sharpe_after_all_costs |   tstat_after_all_costs |   event_count |   event_tstat |
|:---------------------------------------------------------|-----------------------------:|--------------------:|--------------------:|-----------------------------------:|-------------------------:|------------------------:|--------------:|--------------:|
| neg_z2_root1.5_exit0.25_mdq90_normq50_all_topo_hold      |                       0.3526 |              0.0817 |              0.0663 |                             0.0663 |                   0.6887 |                  2.7578 |           148 |        3.6103 |
| neg_z2_root1.5_exit0.25_mdq90_normq50_all_topo_exit      |                       0.3418 |              0.0817 |              0.0651 |                             0.0651 |                   0.6695 |                  2.6811 |           148 |        3.4553 |
| neg_z2_root1.5_exit0.5_mdq90_normq50_all_topo_hold       |                       0.3324 |              0.0817 |              0.0663 |                             0.0663 |                   0.6548 |                  2.6221 |           148 |        3.4038 |
| neg_z2_root1.5_exit0.5_mdq90_normq50_all_topo_exit       |                       0.3312 |              0.0817 |              0.0651 |                             0.0651 |                   0.6525 |                  2.6129 |           148 |        3.3854 |
| neg_z2.5_root1.5_exit0.25_mdq90_normq50_all_topo_hold    |                       0.3132 |              0.0682 |              0.0488 |                             0.0489 |                   0.6539 |                  2.6187 |           123 |        3.3499 |
| neg_z2_root1.5_exit0.25_mdq90_normq50_post2020_topo_hold |                       0.3062 |              0.0301 |              0.0663 |                             0.0663 |                   0.7476 |                  2.9937 |            56 |        3.9634 |
| neg_z2_root1.5_exit0.25_mdq90_normq50_post2020_topo_exit |                       0.3050 |              0.0301 |              0.0651 |                             0.0651 |                   0.7447 |                  2.9823 |            56 |        3.9294 |
| neg_z2.5_root1.5_exit0.25_mdq90_normq50_all_topo_exit    |                       0.3024 |              0.0682 |              0.0476 |                             0.0477 |                   0.6335 |                  2.5367 |           123 |        3.1934 |
| neg_z2.5_root1.5_exit0.5_mdq90_normq50_all_topo_hold     |                       0.2929 |              0.0682 |              0.0488 |                             0.0489 |                   0.6179 |                  2.4743 |           123 |        3.1386 |
| neg_z2_root1.5_exit0.5_mdq90_normq50_post2020_topo_hold  |                       0.2925 |              0.0302 |              0.0663 |                             0.0663 |                   0.7205 |                  2.8851 |            56 |        3.8020 |
| neg_z2.5_root1.5_exit0.5_mdq90_normq50_all_topo_exit     |                       0.2917 |              0.0682 |              0.0476 |                             0.0477 |                   0.6154 |                  2.4644 |           123 |        3.1200 |
| neg_z2_root1.5_exit0.5_mdq90_normq50_post2020_topo_exit  |                       0.2913 |              0.0302 |              0.0651 |                             0.0651 |                   0.7176 |                  2.8736 |            56 |        3.7691 |

## Event Stats

| variant                                             | root   |   events |   mean_event_return |   median_event_return |   event_tstat |   win_rate |   mean_duration_days |
|:----------------------------------------------------|:-------|---------:|--------------------:|----------------------:|--------------:|-----------:|---------------------:|
| neg_z2_root1.5_exit0.25_mdq90_normq50_all_topo_hold | PL     |       48 |              0.0063 |                0.0046 |        2.0198 |     0.7083 |               1.7500 |
| neg_z2_root1.5_exit0.25_mdq90_normq50_all_topo_hold | HG     |       55 |              0.0056 |                0.0013 |        1.8802 |     0.5818 |               1.5273 |
| neg_z2_root1.5_exit0.25_mdq90_normq50_all_topo_hold | SI     |       45 |              0.0052 |                0.0041 |        2.9655 |     0.7556 |               1.5111 |

## Files

- `strategy_metrics.csv`
- `split_metrics.csv`
- `event_log.csv`
- `event_summary.csv`
- `best_strategy_returns.csv`
- `best_strategy_equity.png`
- `top_variant_metrics.png`
