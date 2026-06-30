# HYP-0024-metals-1m-ewma-event-exit: Metals 1-minute EWMA residual entries with event-driven exits

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Selected EWMA event-exit variant was positive in test but failed full-sample robustness after costs.

## Hypothesis

The metal complex may contain short-horizon relative-value dislocations that are
better handled as state-contingent events than as fixed-horizon bets. A
1-minute EWMA conditional residual should define the entry state, while exits
should occur when the residual has decayed, crossed through fair value, exceeded
a residual stop, or the mark is invalid.

## Data

- 1-minute continuous futures marks from
  `/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/continuous`.
- Per-side cost estimates from the prior corrected metals flow experiment.
- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.
- Sample: `2023-06-22T00:00:00Z` through `2026-06-22T00:00:00Z`.

The local tree does not currently contain full 1-minute continuous bars for CL
or rates. This test is therefore the 1-minute metals implementation of the
entry/exit mechanism, not the full metals + CL + rates universe.

## Test

For each timestamp:

1. Estimate EWMA conditional fair-value residuals from 1-minute log prices.
2. Standardize residuals using lagged EWMA residual mean and volatility.
3. Enter a market-neutral residual fade when `abs(z)` breaches the entry
   threshold.
4. Exit without a fixed holding horizon when any event fires:
   residual decay to the exit band, residual sign flip, residual stop, or
   invalid residual.
5. Charge per-side transaction costs on root-level turnover.

The parameter grid is selected only on the training sample. The selected
variant is evaluated on the embargoed held-out sample, starting flat.

## Decision Rule

Promote only if the selected held-out variant has positive net return after
costs, gross/cost above 3x, and bar-level annualized t-stat at least 1.65.
Otherwise reject or revise the mechanism.

## Conceptual Description

The metal complex may contain short-horizon relative-value dislocations that are better handled as state-contingent events than as fixed-horizon bets. A 1-minute EWMA conditional residual should define the entry state, while exits should occur when the residual has decayed, crossed through fair value, exceeded a residual stop, or the mark is invalid.

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
