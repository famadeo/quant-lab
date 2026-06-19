# HYP-0004-xap-xab-longer-1m: Longer-window XAP-XAB pairs robustness test

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `revise`
- Owner: `famadeo`
- Decision notes: Forced backtest after strict longer-window selection failure. No trading claim.

## Hypothesis

The positive XAP-XAB result from the short 2026 GLBX equity-futures run may be a local sector-rotation effect. A longer pair-specific sample should reveal whether the z-score and Mahalanobis rules continue to generate enough gross return to clear costs.

## Conceptual Description

This is a robustness test for the XAP-XAB pair after it stood out in the shorter equity-futures screen. The objective is to separate a persistent relationship from a local selection artifact by extending the sample across 2024-2026 and then comparing the forced pair result with the strict selection outcome. A failure here weakens the case that the earlier result reflected a durable edge.

## Experiment Design

- Roots: `XAP, XAB`
- Asset groups: SectorIndex (2)
- Pair scope: `intra_asset_class`
- Lookback: `390` bars
- Signal lag: `1` bars
- Rebalance interval: `30` bars
- Selection enabled: `True`
- Train fraction: `0.6`
- Fee bps: `0.5`
- Slippage bps: `1.0`

## Strict Selection

| Pair | Selected | Reason | Observations | ADF p-value | Half-life bars |
|---|---:|---|---:|---:|---:|
| XAP-XAB | False | spread_not_stationary | 31,885 | 0.20 | 1513.39 |

## Results: Forced Results

- Completed at: `2026-06-19T12:48:17.005783+00:00`
- Candidate pairs: `1`
- Selected pairs: `1`

| Method | Total Return | Sharpe | Max Drawdown | Active Fraction | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Mahalanobis | -5.01% | -3.95 | -7.28% | 63.07% | 179.23 | 205 |
| Z-score | 6.83% | 7.14 | -4.78% | 44.22% | 67.98 | 85 |
| Z-score + Mahalanobis filter | 3.33% | 6.02 | -2.43% | 16.70% | 29.57 | 35 |

### Selection Reasons

| Key | Value |
|---|---:|
| selection_disabled | 1 |

### Top Pairs By Sharpe

| Pair | Method | Asset Class | Total Return | Sharpe | Trades |
|---|---|---|---:|---:|---:|
| XAP-XAB | Z-score | SectorIndex | 6.83% | 7.14 | 85 |
| XAP-XAB | Z-score + Mahalanobis filter | SectorIndex | 3.33% | 6.02 | 35 |
| XAP-XAB | Mahalanobis | SectorIndex | -5.01% | -3.95 | 205 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
