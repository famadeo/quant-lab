# HYP-0015 Corrected Metals Flow-Filtered Residual Reversion

## Purpose

Re-test the HYP-0014 metals flow-filtered residual mean-reversion result after
correcting the data construction issue found in review.

## Correction

- Flow features are built from all outright trades across GC, SI, HG, PL, and PA.
- Traded returns and residual z-scores are built from the existing 1-minute
  roll-adjusted continuous metals panel, sampled as-of cross-sectional dollar
  bar end times.
- Outright endpoint symbols are retained as diagnostics, but they are not used
  to compute traded returns.
- Price marks are masked when the continuous series is stale or near an active
  contract switch.
- Flow thresholds use shifted rolling quantiles.
- Trade-size bucket thresholds are calibrated from only the first 30 days of
  the sample.

## Decision Rule

The HYP-0014 claim is only rehabilitated if the corrected run remains positive
after costs, survives cost stress, has acceptable drawdown, and produces
credible walk-forward/purged split behavior without relying on mixed-contract
return artifacts.
