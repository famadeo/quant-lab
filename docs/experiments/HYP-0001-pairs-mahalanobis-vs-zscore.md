# HYP-0001-pairs-mahalanobis-vs-zscore: Mahalanobis pairs trading versus z-score pairs trading

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `revise`
- Owner: `famadeo`
- Decision notes: Initial Mahalanobis versus z-score comparison. No trading claim.

## Hypothesis

For intra-sector futures pairs, a Mahalanobis-distance mean-reversion signal can identify joint deviations more effectively than a scalar spread z-score because it accounts for pair covariance and direction in the two-asset state space.

## Conceptual Description

This experiment asks whether pair dislocations are better described as a one-dimensional spread becoming unusually wide, or as a two-asset joint state becoming unusual relative to its historical covariance. The z-score method treats the pair as a scalar spread; the Mahalanobis method treats the same pair as a covariance-aware outlier problem. Both methods are tested on the same intra-sector futures universe so differences come from signal construction, turnover, and costs rather than from pair selection.

## Experiment Design

- Roots: `SR3, ZQ, ZT, ZF, ZN, TN, ZB, UB, IQB, HYB, DHB, DHY, ES, NQ, YM, RTY, NIY, 6E, 6J, 6B, 6C, 6A, 6N, 6S, 6M, 6L, CNH, 6Z, CL, BZ, +23 more`
- Asset groups: Rates (8), Credit (4), Equity (5), FX (11), Energy (6), Metals (8), Agriculture (11)
- Pair scope: `intra_asset_class`
- Lookback: `96` bars
- Signal lag: `1` bars
- Rebalance interval: `12` bars
- Selection enabled: `True`
- Train fraction: `0.6`
- Fee bps: `0.5`
- Slippage bps: `1.0`

## Results

- Completed at: `2026-06-18T22:26:35.459637+00:00`
- Candidate pairs: `179`
- Selected pairs: `50`

| Method | Total Return | Sharpe | Max Drawdown | Active Fraction | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Mahalanobis | -1.06% | -11.25 | -1.19% | 98.37% | 30.61 | 883 |
| Z-score | -0.50% | -9.63 | -0.57% | 99.96% | 39.84 | 1,211 |

### Selection Reasons

| Key | Value |
|---|---:|
| insufficient_train_observations | 19 |
| low_average_pair_volume | 1 |
| low_return_correlation | 24 |
| selected | 50 |
| spread_not_stationary | 85 |

### Top Pairs By Sharpe

| Pair | Method | Asset Class | Total Return | Sharpe | Trades |
|---|---|---|---:|---:|---:|
| 6J-6C | Mahalanobis | FX | 0.16% | 14.43 | 3 |
| ZC-KE | Z-score | Agriculture | 3.46% | 10.55 | 33 |
| NQ-NIY | Z-score | Equity | 2.70% | 9.39 | 34 |
| ZC-ZW | Z-score | Agriculture | 2.33% | 7.30 | 31 |
| SI-PL | Z-score | Metals | 4.05% | 6.38 | 39 |
| 6J-CNH | Mahalanobis | FX | 0.04% | 6.09 | 2 |
| 6E-6L | Z-score | FX | 0.79% | 5.66 | 39 |
| 6S-6L | Z-score | FX | 0.72% | 5.20 | 36 |
| ZS-KE | Mahalanobis | Agriculture | 1.61% | 4.98 | 55 |
| 6J-6L | Z-score | FX | 0.48% | 4.94 | 34 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
