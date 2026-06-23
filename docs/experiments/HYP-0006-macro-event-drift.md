# HYP-0006-macro-event-drift: Post-macro-event futures drift/reversal

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Post-5-minute macro-event continuation/reversal lost money after costs.

## Hypothesis

After scheduled US macro releases, the first five minutes of price reaction can
be used to select either continuation or reversal for the next 55 minutes.

## Method

For each root, fit the relationship between the event-to-5-minute return and the
5-to-60-minute forward return on the training events only. In the test set, trade
the sign implied by the training slope. Costs are 3 bps round trip per event leg.

## Decision Rule

Reject unless the equal-weight event portfolio has positive net return, event
t-statistic of at least 1.65, and at least 60% positive roots out of sample.

## Conceptual Description

After scheduled US macro releases, the first five minutes of price reaction can be used to select either continuation or reversal for the next 55 minutes.

## Experiment Design

- Roots: `ZT, ZF, ZN, ZB, SR3, ES, NQ, 6E, 6J, CL`
- Asset groups: `n/a`
- Pair scope: `n/a`
- Lookback: `n/a` bars
- Signal lag: `n/a` bars
- Rebalance interval: `n/a` bars
- Selection enabled: `n/a`
- Train fraction: `0.6`
- Fee bps: `3.0`
- Slippage bps: `n/a`

## Results

No result artifact was available when the wiki was built.

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
