# HYP-0005: Volatility-Clock Low-Size OFI Predictive Test

## Hypothesis

Low-size trade order-flow imbalance, measured on per-asset volatility-clock bars,
predicts the next volatility-bar return for the candidate roots that looked most
interesting in exploration.

The primary claim is tested on `ofi_low`. `ofi_high` and `ofi_spread` are
reported as controls, not as the preregistered claim.

## Universe

The test is limited to the five roots selected from the exploratory volatility
clock notebook:

- `SR3`
- `ZT`
- `ZB`
- `CL`
- `RTY`

This is not a broad discovery run. It is a follow-up validation of a specific
lead.

## Data

- Source: cached Databento `GLBX.MDP3` 5-minute continuous returns and raw
  `trades` data.
- Window: `2026-05-17T00:00:00Z` to `2026-06-17T00:00:00Z`, covering the
  `2026-05-17` to `2026-06-16` analysis period.
- Session: `13:30` to `20:00` UTC.
- Aggressor side: `B` is buy-aggressor and `A` is sell-aggressor.

## Method

For each root:

1. Split the available RTH trading dates chronologically into train and test
   sets using the first 60% of dates for training.
2. Estimate the low and high trade-size thresholds only on training-period RTH
   outright trades.
3. Aggregate signed low-size and high-size flow into 5-minute bars.
4. Estimate the volatility-clock variance threshold only on training-period
   5-minute returns. The target is six average training 5-minute bars of
   realized variance.
5. Build day-reset volatility bars from 5-minute returns and bucketed OFI.
6. Fit a training-only one-factor slope of next-volatility-bar return on each
   OFI feature.
7. In the test period, trade the next volatility bar using the sign of the
   training slope and the previous completed bar's clipped OFI value.

The strategy has a one-volatility-bar execution lag. Same-bar returns are never
used for the trading result.

## Costs

The net return subtracts `1.5` bps per unit of position turnover. A move from
flat to a full-size position costs `1.5` bps; a full reversal costs `3.0` bps.

## Falsification Criteria

The primary `ofi_low` claim fails if any of these hold out of sample:

- Pooled net return across the five roots is not positive.
- Pooled event t-statistic is below `1.65`.
- Fewer than 60% of roots have positive net return.

If it passes, the status remains `revise`, not `paper_trade`, because the sample
is short and the candidate roots were selected after prior exploration.

## Result

Initial run completed on 2026-06-22. The primary `ofi_low` claim was rejected:

- Pooled out-of-sample net return: `-0.73%`.
- Pooled event t-statistic: `-2.43`.
- Positive-root fraction: `20%`.

The only positive `ofi_low` root was `RTY`; `SR3`, `ZT`, `ZB`, and `CL` were
negative after costs.
