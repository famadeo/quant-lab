# HYP-0010-futures-carry-momentum: Futures carry plus momentum baseline

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Carry, momentum, and combined daily portfolios were negative after costs.

## Hypothesis

A cross-sectional futures portfolio combining term-structure carry and 12-month
momentum has positive out-of-sample net return without needing fast execution.

## Method

Use daily per-contract futures data from 2010-2024. Carry is approximated as the
log front/next-contract close ratio using the active contract from the
roll-adjusted continuous series. Momentum is the 252-day continuous log-price
change. Each factor is cross-sectionally standardized; the combined strategy is
the average of carry and momentum scores.

## Decision Rule

Reject unless the combined out-of-sample portfolio has positive net return and a
daily event t-statistic of at least 1.65 after turnover costs.

## Conceptual Description

A cross-sectional futures portfolio combining term-structure carry and 12-month momentum has positive out-of-sample net return without needing fast execution.

## Experiment Design

- Roots: `6A, 6B, 6C, 6E, 6J, CL, ES, GC, HG, NQ, SI, ZT, ZF, ZN, ZB`
- Asset groups: `n/a`
- Pair scope: `n/a`
- Lookback: `252` bars
- Signal lag: `n/a` bars
- Rebalance interval: `n/a` bars
- Selection enabled: `n/a`
- Train fraction: `0.7`
- Fee bps: `1.5`
- Slippage bps: `n/a`

## Results

No result artifact was available when the wiki was built.

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
