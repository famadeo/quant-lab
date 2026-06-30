# HYP-0017 Corrected Metals Flow Research Framework

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
