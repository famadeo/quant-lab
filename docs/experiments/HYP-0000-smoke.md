# HYP-0000-smoke: Synthetic moving-average smoke test

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `revise`
- Owner: `famadeo`
- Decision notes: Smoke run only. No trading claim.

## Hypothesis

This experiment validates that the research stack can run a deterministic signal, lag positions, apply costs, write artifacts, and log to MLflow.

## Conceptual Description

This is an infrastructure-control experiment, not a market hypothesis. It uses synthetic data to confirm that the lab can execute the full scientific-method workflow end to end before any real trading idea is evaluated.

## Experiment Design

- Roots: `n/a`
- Asset groups: `n/a`
- Pair scope: `n/a`
- Lookback: `n/a` bars
- Signal lag: `n/a` bars
- Rebalance interval: `n/a` bars
- Selection enabled: `n/a`
- Train fraction: `n/a`
- Fee bps: `0.5`
- Slippage bps: `1.0`

## Results

- Completed at: `2026-06-18T22:27:39.738710+00:00`

| Metric | Value |
|---|---:|
| annualized_return | -0.07 |
| annualized_volatility | 0.09 |
| average_exposure | 0.33 |
| max_drawdown | -0.25 |
| observations | 783 |
| sharpe_ratio | -0.88 |
| total_return | -0.22 |
| total_turnover | 23.00 |
| trades | 23 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
