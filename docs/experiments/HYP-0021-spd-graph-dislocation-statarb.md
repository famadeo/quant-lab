# HYP-0021-spd-graph-dislocation-statarb: SPD-regime graph-cluster dislocation stat-arb

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `research`
- Owner: `famadeo`
- Decision notes: Pre-registered geometry stack using SPD regimes, standardized-sphere graph structure, and residual-cloud dislocation trades.

## Hypothesis

Equity residual dislocations are only exploitable when conditioned on market
geometry. In stable covariance regimes, persistent MST/PMFG graph clusters should
mean-revert after name-level residual-cloud dislocations. In transition regimes,
the same dislocations may continue or should be skipped.

## Data

- Source project: `/home/famadeo/archive/SP500_statarb`.
- Prices: adjusted daily closes.
- Baseline comparator:
  `/home/famadeo/archive/SP500_statarb/data/backtests/graph_pca_residual/metrics.json`.
- Initial sample: 2021-05-24 through 2026-05-20 to match the archived
  graph/PCA residual baseline.

## Test

Use rolling windows only. At each rebalance date:

1. Estimate a shrinkage SPD correlation matrix from the trailing return window.
2. Map the SPD matrix to log coordinates and classify the covariance regime from
   trailing-only features.
3. Build a standardized-sphere correlation graph using MST and, if feasible,
   PMFG or TMFG.
4. Estimate market/sector/PCA/graph-neighbor residuals.
5. Treat the cross-section of scaled residuals as an R^N dislocation cloud.
6. In stable regimes, long negative cloud outliers and short positive cloud
   outliers within persistent graph clusters.
7. In transition regimes, test both skip and continuation variants.

## Baselines

- Archived `graph_pca_ungated`.
- Archived `graph_pca_gated`.
- Archived `pca_only_gated`.
- Sector-neutral residual reversal from the archived strategy battery.

## Decision Rule

Promote only if the selected walk-forward variant has positive out-of-sample net
return after 5 bps costs, gross/cost above 3x, daily t-stat above 2.0 or bootstrap
p-value below 5%, OOS Sharpe above 1.0, and drawdown below 15%. It must also beat
the archived graph/PCA residual baselines.

## Conceptual Description

Equity residual dislocations are only exploitable when conditioned on market geometry. In stable covariance regimes, persistent MST/PMFG graph clusters should mean-revert after name-level residual-cloud dislocations. In transition regimes, the same dislocations may continue or should be skipped.

## Experiment Design

- Roots: `n/a`
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

No result artifact was available when the wiki was built.

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
