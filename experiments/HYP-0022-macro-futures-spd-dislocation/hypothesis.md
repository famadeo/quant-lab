# HYP-0022: Macro-futures SPD-regime residual dislocation

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
