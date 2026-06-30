# HYP-0022-macro-futures-spd-dislocation: Macro-futures SPD-regime residual dislocation

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Training-selected daily macro-futures SPD residual strategy failed held-out validation and did not beat the ungated control.

## Hypothesis

In a liquid macro-futures universe of metals, crude oil, and the Treasury curve,
cross-asset residual dislocations should be exploitable only when market geometry
is stable. The trade should fade residual-cloud outliers during stable SPD
covariance regimes and avoid or separately test continuation during high-velocity
covariance transitions.

## Data

- Daily roll-adjusted continuous futures from
  `/home/famadeo/research/databento-asset-browser/data/futures_continuous`.
- Initial powered universe: `GC`, `SI`, `HG`, `PL`, `PA`, `CL`, `ZT`, `ZF`,
  `ZN`, `ZB`.
- Rates are sign-normalized into yield/risk space for regime and residual
  estimation, then converted back to futures-price exposure for PnL.
- `TN`, `UB`, `SR3`, and `ZQ` are deferred to the short 5-minute extension
  because they are not present in the longer daily continuous CSV set.

## Test

At each date, using only trailing data:

1. Build a risk-normalized return panel.
2. Estimate rolling SPD correlation geometry and MST edge persistence.
3. Classify stable and transition regimes from rolling quantiles of SPD
   velocity and edge persistence.
4. Estimate rolling macro-factor residuals in risk-normalized log-price space.
5. Convert residuals to z-scores and build sparse dislocation-cloud positions.
6. Compare ungated reversion, stable-regime reversion, non-transition reversion,
   and stable-reversion plus transition-continuation.

The variant grid is selected only on the training split. The selected variant is
then evaluated on the held-out test split.

## Decision Rule

Promote only if the selected held-out variant has positive net return, gross/cost
above 3x, daily t-stat at least 1.65, and beats the ungated residual-reversion
control after transaction costs.

## Conceptual Description

In a liquid macro-futures universe of metals, crude oil, and the Treasury curve, cross-asset residual dislocations should be exploitable only when market geometry is stable. The trade should fade residual-cloud outliers during stable SPD covariance regimes and avoid or separately test continuation during high-velocity covariance transitions.

## Experiment Design

- Roots: `GC, SI, HG, PL, PA, CL, ZT, ZF, ZN, ZB`
- Asset groups: `n/a`
- Pair scope: `n/a`
- Lookback: `n/a` bars
- Signal lag: `n/a` bars
- Rebalance interval: `n/a` bars
- Selection enabled: `n/a`
- Train fraction: `0.7`
- Fee bps: `1.5`
- Slippage bps: `n/a`

## Results

No result artifact was available when the wiki was built.

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
