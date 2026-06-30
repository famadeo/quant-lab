# HYP-0018-metals-flow-corrected-research-framework-3y-core: Corrected three-year core metals cross-sectional flow research framework

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `research`
- Owner: `famadeo`
- Decision notes: Primary multi-year corrected framework sample; five-root core data quality is strong, standalone flow ICs remain below 0.03, and q99 flow states support PA/PL event-filter research.

## Hypothesis

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

## Conceptual Description

On the longer corrected three-year core metals dataset, do cross-sectional flow anomalies provide robust predictive information beyond the 12-month ALI-inclusive extension?

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

- Completed at: `2026-06-24T19:26:24.841844+00:00`

No publishable method-level metrics were available in this result artifact.

### Summary Highlights

| Metric | Value |
|---|---:|
| `start` | 2023-06-22T00:00:00Z |
| `end` | 2026-06-22T00:00:00Z |
| `primary_complete_bars` | 242,166 |
| `cointegration_pairs_p_lt_0_05` | 1 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
