# HYP-0023: 5-minute macro-futures SPD/flow dislocation

## Hypothesis

In the metals + crude + rates futures complex, intraday residual dislocations are
only exploitable when both covariance geometry and order-flow state agree. In
stable SPD regimes, residual outliers created or reinforced by same-direction
trade flow should mean-revert over subsequent 5-minute bars. In high-velocity SPD
transition regimes, the same flow-confirmed dislocations may continue or should
be skipped.

## Data

- 5-minute continuous futures bars from
  `/home/famadeo/research/databento-asset-browser/data/futures_5m`.
- 5-minute trade-flow features from
  `/home/famadeo/research/databento-asset-browser/data/trades`.
- Universe: `GC`, `SI`, `HG`, `PL`, `PA`, `CL`, `SR3`, `ZQ`, `ZT`, `ZF`, `ZN`,
  `TN`, `ZB`, `UB`.
- Rates are sign-normalized into yield/risk space for regimes, residuals, and
  flow features, then converted back to futures-price exposure for PnL.

## Test

Use only aligned bars where returns and flow are present for the full universe.
For each timestamp:

1. Estimate rolling SPD correlation state from risk-normalized 5-minute returns.
2. Compute MST edge persistence and classify stable/transition regimes.
3. Estimate rolling macro-factor residuals from risk-normalized log-price space.
4. Convert residuals and order flow to rolling z-scores.
5. Compare residual-only, flow-confirmed residual reversion, transition flow
   continuation, flow-only momentum, and flow-only impact-decay controls.

The variant grid is selected only on the training dates. The selected variant is
then evaluated on held-out dates.

## Decision Rule

This is a short-sample mechanism test. Promote only if the selected held-out
variant has positive net return after costs, gross/cost above 3x, bar-level
t-stat at least 1.65, and beats the corresponding ungated residual-reversion
control.
