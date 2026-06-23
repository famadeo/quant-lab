# HYP-0009: Intraday Trend Following With Flow Confirmation

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
