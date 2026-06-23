# HYP-0009-trend-flow-filter: Intraday trend following with flow confirmation

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Flow-filtered intraday trend underperformed baseline after costs.

## Hypothesis

A simple intraday trend signal improves when it is only traded in bars where
block-flow OFI confirms the trend and volume is not unusually low.

## Method

Use 5-minute RTH futures returns and cached 5-minute flow bars. The baseline
trades the sign of the prior 12-bar return when its absolute value exceeds the
training 60th percentile. The filtered version also requires same-sign
`ofi_block` and at least median training volume. Signals trade the next bar.

## Decision Rule

Reject unless the filtered trend portfolio is net positive, has event t-statistic
of at least 1.65, and beats the baseline trend net return.

## Conceptual Description

A simple intraday trend signal improves when it is only traded in bars where block-flow OFI confirms the trend and volume is not unusually low.

## Experiment Design

- Roots: `SR3, ZT, ZF, ZN, ZB, ES, NQ, YM, RTY, 6E, 6J, 6B, 6C, 6A, CL, BZ, GC, SI`
- Asset groups: `n/a`
- Pair scope: `n/a`
- Lookback: `n/a` bars
- Signal lag: `n/a` bars
- Rebalance interval: `n/a` bars
- Selection enabled: `n/a`
- Train fraction: `0.6`
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
