# HYP-0017-metals-flow-corrected-research-framework-12m: Corrected 12-month metals cross-sectional flow research framework

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `research`
- Owner: `famadeo`
- Decision notes: Full six-root corrected framework rerun; ALI is sparse/stale, standalone flow features remain weak, and relative-value residual states dominate pure flow signals.

## Hypothesis

## Question

Using the new full 12-month metals input dataset, do cross-sectional trade-flow anomalies across `GC`, `SI`, `HG`, `PL`, `PA`, and `ALI` predict corrected future returns or residual dislocations?

## Correction Versus HYP-0012

HYP-0012 originally used root-level last trade prices at bar endpoints. After the HYP-0015 data review, that is no longer acceptable for return or fair-value work because it can mix outright contracts around rolls.

This rerun keeps the HYP-0012 research framework but changes the price path:

- trade flow comes from all outright trades;
- returns and fair-value residuals come from roll-adjusted 1-minute continuous marks;
- stale continuous marks and roll-adjacent bars are masked;
- price-validity artifacts are written with the run.

## Universe

- GC
- SI
- HG
- PL
- PA
- ALI

## Window

`2025-06-23` through `2026-06-22`, the common overlap of the full six-root trade set and continuous marks.

## Decision Criteria

The framework is exploratory, not directly a strategy pass/fail. A tradable lead must still satisfy:

- corrected data construction;
- stable signal sign across roots and horizons;
- interpretable economic mechanism;
- gross return at least 3x estimated costs in a subsequent executable backtest.

## Conceptual Description

Using the new full 12-month metals input dataset, do cross-sectional trade-flow anomalies across `GC`, `SI`, `HG`, `PL`, `PA`, and `ALI` predict corrected future returns or residual dislocations?

## Experiment Design

- Roots: `GC, SI, HG, PL, PA, ALI`
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

- Completed at: `2026-06-24T19:05:50.582967+00:00`

No publishable method-level metrics were available in this result artifact.

### Summary Highlights

| Metric | Value |
|---|---:|
| `start` | 2025-06-23T00:00:00Z |
| `end` | 2026-06-22T00:00:00Z |
| `primary_complete_bars` | 115,704 |
| `cointegration_pairs_p_lt_0_05` | 0 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
