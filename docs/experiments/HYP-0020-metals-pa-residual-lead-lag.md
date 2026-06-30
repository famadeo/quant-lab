# HYP-0020-metals-pa-residual-lead-lag: PA residual lead-lag into liquid metals

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `revise`
- Owner: `famadeo`
- Decision notes: Candidate pattern passed train/test gates, but it came from iterative search and needs forward validation before any trading claim.

## Hypothesis

On corrected three-year core metals data, an extreme `PA` fair-value residual
state leads liquid metals returns. When `PA` is rich or cheap relative to the
EWMA metals fair-value model, `GC` and `SI` continue in the same residual-sign
direction over the next 50 cross-sectional dollar bars.

## Data

- Trade flow and cross-sectional dollar bars come from the corrected HYP-0018
  three-year core metals artifacts.
- Returns are computed from roll-adjusted 1-minute continuous marks, not mixed
  outright endpoint prices.
- Transaction costs use the HYP-0015 MBP1 median half-spread estimates.

## Test

Use the first 70% of bars only to screen candidate signals and direction. Apply
the selected rule to the full sample with a 50-bar embargo before the test
split:

- Signal: `PA` fair-value residual z-score.
- Entry: `abs(z) >= 1.5`.
- Direction: same sign as `PA` residual.
- Holding period: 50 cross-sectional dollar bars.
- Target portfolio: 50% `GC`, 50% `SI`.
- Costs: per-side MBP1 half-spread on turnover.

## Decision Rule

This is a candidate pattern, not a live-trading approval. It passes this
experiment only if train and purged-test net returns are positive, purged-test
gross/cost is above 3x, and full-sample t-stat is above 2.0 after costs.

## Conceptual Description

On corrected three-year core metals data, an extreme `PA` fair-value residual state leads liquid metals returns. When `PA` is rich or cheap relative to the EWMA metals fair-value model, `GC` and `SI` continue in the same residual-sign direction over the next 50 cross-sectional dollar bars.

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
