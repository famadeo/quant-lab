# HYP-0014-metals-flow-filtered-residual-reversion-3y: Three-year core metals flow-filtered residual mean-reversion strategy

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `invalid_data_construction`
- Owner: `famadeo`
- Decision notes: Invalidated after review; about 85% of net PnL was on mixed-contract endpoint switch bars.

## Hypothesis

The HYP-0013 flow-filtered residual mean-reversion result remains positive over
a longer three-year core-metals sample when `ALI` is excluded and the traded
universe is restricted to `GC`, `SI`, `HG`, `PL`, and `PA`.

## Test

Use the same strategy parameters as HYP-0013, but replace the 12-month trade
sample with a three-year core-only tick-trade sample from `2023-06-22` through
`2026-06-22`.

## Decision Standard

The longer sample should remain positive after base MBP-1 spread costs, clear at
least three times transaction costs under stress, and avoid concentrating all
profit in the original 12-month validation window.

## Conceptual Description

The HYP-0013 flow-filtered residual mean-reversion result remains positive over a longer three-year core-metals sample when `ALI` is excluded and the traded universe is restricted to `GC`, `SI`, `HG`, `PL`, and `PA`.

## Experiment Design

- Roots: `GC, SI, HG, PL, PA`
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

- Completed at: `2026-06-24T15:16:09.372420+00:00`

| Method | Total Return | Sharpe | Max Drawdown | Active Fraction | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| cost_0.0 | 402.67% | 5.69 | -7.37% | 13.89% | 1527.17 |  |
| cost_1.0 | 367.02% | 5.21 | -7.50% | 13.89% | 1527.17 |  |
| cost_2.0 | 331.36% | 4.72 | -7.63% | 13.89% | 1527.17 |  |
| cost_3.0 | 295.71% | 4.22 | -7.76% | 13.89% | 1527.17 |  |

### Summary Highlights

| Metric | Value |
|---|---:|
| `start` | 2023-06-22T00:00:00Z |
| `end` | 2026-06-22T00:00:00Z |
| `primary_complete_bars` | 242,162 |
| `filter_pass_rate` | 0.03 |
| `fdr_significant_tests` | 293 |
| `selected_net_return` | 3.67 |
| `selected_gross_to_cost` | 11.29 |
| `selected_tstat` | 9.02 |
| `selected_max_drawdown` | -0.07 |
| `selected_active_bars` | 33,646 |
| `selected_turnover` | 1527.17 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
