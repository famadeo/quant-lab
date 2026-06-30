# HYP-0018 Corrected Three-Year Core Metals Flow Framework

## Question

On the longer corrected three-year core metals dataset, do cross-sectional flow anomalies provide robust predictive information beyond the 12-month ALI-inclusive extension?

## Universe

- GC
- SI
- HG
- PL
- PA

ALI is excluded because three-year outright trade data is not available in the current local dataset.

## Correction Standard

This run uses the corrected HYP-0015 data construction:

- all outright trades for flow features;
- roll-adjusted 1-minute continuous marks for returns and fair-value residuals;
- stale-price and roll-adjacent bar masks;
- existing corrected HYP-0015 cached bars where available.

## Role In Research Stack

This is the primary robustness framework sample. HYP-0017 remains the shorter six-root extension for ALI/liquidity diagnostics.
