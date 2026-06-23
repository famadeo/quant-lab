# HYP-0007-vol-clock-impact-decay: Volatility-clock impact decay after extreme high-size OFI

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Positive pooled net result but weak t-stat and only 40% positive roots.

## Hypothesis

Extreme high-size OFI combined with a large same-bar price move contains
next-volatility-bar information. The training set decides whether each root is a
continuation or fade market.

## Method

Use completed volatility bars from `HYP-0005`. An event requires both absolute
`ofi_high` and absolute same-bar return to exceed their training 80th percentile,
with OFI and return signs aligned. The test trades the next volatility bar.

## Decision Rule

Reject unless pooled net return is positive, event t-statistic is at least 1.65,
and at least 60% of roots are positive after costs.

## Conceptual Description

Extreme high-size OFI combined with a large same-bar price move contains next-volatility-bar information. The training set decides whether each root is a continuation or fade market.

## Experiment Design

- Roots: `SR3, ZT, ZB, CL, RTY`
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
