# HYP-0002-equity-top100-pairs: Equity-only relative value branch

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `revise`
- Owner: `famadeo`
- Decision notes: Strict equity-index futures baseline. Top-100 market-cap branch not run until billable Databento equity pull is explicitly allowed.

## Hypothesis

Equity-only pairs may contain more idiosyncratic relative-value structure than the cross-asset macro futures universe. The strict futures baseline uses the locally available equity-index futures; the top-100 market-cap branch uses Databento US equities rather than futures.

## Conceptual Description

This branch narrows the search from a broad macro futures universe to equity relative value. The premise is that equity instruments may share enough common market exposure for spreads to mean-revert, while still retaining enough idiosyncratic dispersion to create tradable relative moves. The documented run is the strict equity-index futures baseline; the top-100 single-name equity branch remains gated behind explicit approval for the billable Databento pull.

## Experiment Design

- Roots: `ES, NQ, YM, RTY, NIY`
- Asset groups: Equity (5)
- Pair scope: `intra_asset_class`
- Lookback: `96` bars
- Signal lag: `1` bars
- Rebalance interval: `12` bars
- Selection enabled: `True`
- Train fraction: `0.6`
- Fee bps: `0.5`
- Slippage bps: `1.0`

## Results

- Completed at: `2026-06-18T22:51:26.075498+00:00`
- Candidate pairs: `10`
- Selected pairs: `2`

| Method | Total Return | Sharpe | Max Drawdown | Active Fraction | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Mahalanobis | -2.71% | -5.07 | -3.95% | 45.73% | 63.89 | 89 |
| Z-score | 2.20% | 7.73 | -0.75% | 52.44% | 39.82 | 66 |

### Selection Reasons

| Key | Value |
|---|---:|
| selected | 2 |
| spread_not_stationary | 8 |

### Top Pairs By Sharpe

| Pair | Method | Asset Class | Total Return | Sharpe | Trades |
|---|---|---|---:|---:|---:|
| NQ-NIY | Z-score | Equity | 2.50% | 6.67 | 40 |
| ES-NIY | Z-score | Equity | 1.89% | 5.77 | 44 |
| NQ-NIY | Mahalanobis | Equity | -2.85% | -4.48 | 68 |
| ES-NIY | Mahalanobis | Equity | -2.59% | -4.74 | 62 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
