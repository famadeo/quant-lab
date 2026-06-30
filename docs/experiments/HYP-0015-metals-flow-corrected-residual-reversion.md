# HYP-0015-metals-flow-corrected-residual-reversion: Corrected three-year metals flow-filtered residual mean-reversion validation

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Corrected run uses all-outright trades for flow and continuous 1-minute marks for returns; base-cost net return was negative and gross/cost was below 1x.

## Hypothesis

## Purpose

Re-test the HYP-0014 metals flow-filtered residual mean-reversion result after
correcting the data construction issue found in review.

## Correction

- Flow features are built from all outright trades across GC, SI, HG, PL, and PA.
- Traded returns and residual z-scores are built from the existing 1-minute
  roll-adjusted continuous metals panel, sampled as-of cross-sectional dollar
  bar end times.
- Outright endpoint symbols are retained as diagnostics, but they are not used
  to compute traded returns.
- Price marks are masked when the continuous series is stale or near an active
  contract switch.
- Flow thresholds use shifted rolling quantiles.
- Trade-size bucket thresholds are calibrated from only the first 30 days of
  the sample.

## Decision Rule

The HYP-0014 claim is only rehabilitated if the corrected run remains positive
after costs, survives cost stress, has acceptable drawdown, and produces
credible walk-forward/purged split behavior without relying on mixed-contract
return artifacts.

## Conceptual Description

Re-test the HYP-0014 metals flow-filtered residual mean-reversion result after correcting the data construction issue found in review.

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

- Completed at: `2026-06-24T17:42:59.638871+00:00`

| Method | Total Return | Sharpe | Max Drawdown | Active Fraction | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| cost_0.0 | 18.26% | 0.50 | -32.05% | 47.41% | 1688.69 |  |
| cost_1.0 | -20.85% | -0.58 | -49.69% | 47.41% | 1688.69 |  |
| cost_10.0 | -372.80% | -9.55 | -373.55% | 47.41% | 1688.69 |  |
| cost_2.0 | -59.95% | -1.65 | -69.44% | 47.41% | 1688.69 |  |
| cost_3.0 | -99.06% | -2.73 | -101.47% | 47.41% | 1688.69 |  |
| cost_5.0 | -177.27% | -4.82 | -178.89% | 47.41% | 1688.69 |  |

### Summary Highlights

| Metric | Value |
|---|---:|
| `start` | 2023-06-22T00:00:00Z |
| `end` | 2026-06-22T00:00:00Z |
| `primary_complete_bars` | 242,166 |
| `filter_pass_rate` | 0.47 |
| `fdr_significant_tests` | 251 |
| `selected_net_return` | -0.21 |
| `selected_gross_to_cost` | 0.47 |
| `selected_tstat` | -1.00 |
| `selected_max_drawdown` | -0.50 |
| `selected_active_bars` | 114,817 |
| `selected_turnover` | 1688.69 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
