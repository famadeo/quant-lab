# HYP-0016-metals-flow-incremental-signal-validation: Incremental signal validation for corrected metals flow features

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Flow terms show weak PA/PL conditional information, but every corrected strategy variant is net negative and gross/cost remains below 0.70x.

## Hypothesis

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

## Conceptual Description

After correcting the HYP-0013/HYP-0014 price-construction error, do cross-sectional flow anomaly variables add statistically and economically meaningful predictive value beyond residual dislocation alone?

## Experiment Design

- Roots: `GC, SI, HG, PL, PA`
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
