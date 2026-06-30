# HYP-0016: Incremental signal validation for corrected metals flow features

## Question

After correcting the HYP-0013/HYP-0014 price-construction error, do cross-sectional flow anomaly variables add statistically and economically meaningful predictive value beyond residual dislocation alone?

## Source

This experiment uses the corrected artifacts from:

`experiments/HYP-0015-metals-flow-corrected-residual-reversion`

The source experiment builds flow features from all outright trades and computes returns/residuals from 1-minute roll-adjusted continuous marks sampled at cross-sectional dollar-bar endpoints.

## Tests

1. Compare residual-reversion strategy variants:
   - residual only
   - residual plus Mahalanobis flow gate
   - residual plus large-small contribution-vector gate
   - residual plus both gates
   - residual plus both gates and root-level large-small confirmation
2. Regress root forward returns on residual reversion signal, flow features, and residual-flow interaction terms using HAC standard errors.
3. Compare signed forward reversion returns for residual extremes with and without flow gates.
4. Generate portfolio-level and root-level signal/forward-return scatter diagnostics.

## Decision Criteria

The flow mechanism needs to clear all of the following before strategy tuning:

- Positive out-of-sample or full-sample net return after current cost estimates.
- Gross return at least three times estimated cost for the realistic-cost variant.
- Flow terms or residual-flow interaction terms with stable sign and significance after multiple-testing adjustment.
- Event-study improvement when flow gates are present versus residual extremes alone.
