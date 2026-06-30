# HYP-0019: Residual-State Momentum Filtered by Extreme Metals Flow Geometry

## Hypothesis

When the metals complex enters an extreme cross-sectional flow-geometry regime, fair-value residual dislocations in the less liquid PGM roots do not immediately converge. Instead, residual states in `PL` and `PA` persist over the next 10-20 cross-sectional dollar bars.

## Economic Mechanism

The expected edge is slow risk transfer. `GC`, `SI`, and `HG` dominate flow and absorb information faster. `PL` and especially `PA` are thinner, so extreme complex-wide flow states can coincide with delayed inventory transfer, wider liquidity provision, and continued repricing before convergence.

## Test

Use the corrected three-year core framework from HYP-0018:

- all-outright trade flow;
- roll-adjusted continuous 1-minute marks for returns;
- shifted walk-forward Mahalanobis thresholds;
- MBP1-derived spread cost estimates;
- train/test split with embargo.

The strategy enters in the direction of the residual z-score and exits on residual decay, sign reversal, maximum holding period, invalid price marks, or stop z-score.

## Success Criteria

A candidate variant should satisfy:

- positive net return after costs;
- gross return at least 3x estimated transaction costs;
- positive purged-test performance;
- economically plausible trade count and holding period.
