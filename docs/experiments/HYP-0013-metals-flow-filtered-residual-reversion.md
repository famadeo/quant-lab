# HYP-0013-metals-flow-filtered-residual-reversion: Flow-filtered metals residual mean-reversion strategy

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `research_pass`
- Owner: `famadeo`
- Decision notes: Full 12-month core metals test using flow anomaly and large-minus-small flow filters; passed gross-to-cost and split/bootstrap diagnostics, pending stricter execution and forward validation.

## Hypothesis

Liquid metals residual dislocations mean-revert more reliably when cross-sectional
flow geometry is abnormal and large-trade flow disagrees with small-trade flow.

## Universe

Core traded roots: `GC`, `SI`, `HG`, `PL`, `PA`.

Isolated diagnostic root: `ALI`, excluded from the first tradable sleeve because
it is much smaller and lacks matching MBP-1 coverage in the current pull.

## Test

Use 12 months of tick trades to build synchronized cross-sectional dollar bars.
Estimate dynamic residual z-scores using no-lookahead EWMA conditional fair value.
Enter residual mean-reversion positions only when EWMA Mahalanobis flow distance
and large-minus-small flow disagreement are both elevated. Exit when residuals
converge toward zero.

## Caveat

The full-sample strategy layer uses EWMA Mahalanobis and EWMA conditional fair
value for runtime stability. The 30-day HYP-0012 notebook remains the heavier
visual diagnostic harness for rolling/expanding/robust variants.

## Conceptual Description

Liquid metals residual dislocations mean-revert more reliably when cross-sectional flow geometry is abnormal and large-trade flow disagrees with small-trade flow.

## Experiment Design

- Roots: `GC, SI, HG, PL, PA, ALI`
- Asset groups: `n/a`
- Pair scope: `n/a`
- Lookback: `n/a` bars
- Signal lag: `n/a` bars
- Rebalance interval: `n/a` bars
- Selection enabled: `n/a`
- Train fraction: `n/a`
- Fee bps: `n/a`
- Slippage bps: `n/a`

## Results

- Completed at: `2026-06-24T02:27:11.898295+00:00`

| Method | Total Return | Sharpe | Max Drawdown | Active Fraction | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| cost_0.0 | 170.23% | 5.98 | -6.81% | 13.93% | 673.21 |  |
| cost_1.0 | 154.62% | 5.44 | -6.98% | 13.93% | 673.21 |  |
| cost_2.0 | 139.00% | 4.90 | -7.15% | 13.93% | 673.21 |  |
| cost_3.0 | 123.39% | 4.36 | -7.32% | 13.93% | 673.21 |  |

### Summary Highlights

| Metric | Value |
|---|---:|
| `start` | 2025-06-22T22:00:00Z |
| `end` | 2026-06-22T00:00:00Z |
| `primary_complete_bars` | 115,649 |
| `filter_pass_rate` | 0.03 |
| `fdr_significant_tests` | 317 |
| `selected_net_return` | 1.55 |
| `selected_gross_to_cost` | 10.90 |
| `selected_tstat` | 5.44 |
| `selected_max_drawdown` | -0.07 |
| `selected_active_bars` | 16,113 |
| `selected_turnover` | 673.21 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
