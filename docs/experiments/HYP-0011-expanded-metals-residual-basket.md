# HYP-0011-expanded-metals-residual-basket: Expanded metals residual mean-reversion basket

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `revise`
- Owner: `famadeo`
- Decision notes: Daily basket passed the initial OOS gate, drawdown controls favored GC/SI/HG/PL, and the 1-minute retest showed the gross signal is swamped by turnover costs.

## Hypothesis

Adding `PL`, `PA`, and `ALI` to the original `GC/SI/HG` metals residual basket
improves robustness by broadening the metals complex while preserving the same
daily relative-value mean-reversion mechanism.

## Data

- Dataset: Databento `GLBX.MDP3`.
- Raw schema: per-contract `ohlcv-1d`.
- Roots: `GC`, `SI`, `HG`, `PL`, `PA`, `ALI`.
- Continuous construction: same monotonic volume-roll, within-contract return
  splice used by the existing daily futures pipeline.
- Start date: `2014-05-06`, because `ALI` history starts there in the pull.

## Method

At each daily close:

1. Compute each root's continuous log price.
2. Subtract the cross-sectional metals mean to form residual log price.
3. Convert residuals to rolling 126-day z-scores.
4. Fade residual z-scores clipped at `+/-2`.
5. Normalize positions to one unit gross exposure.
6. Hold for the next day and rebalance daily.

Costs are `1.5` bps per unit of turnover.

## Decision Rule

Reject unless the out-of-sample portfolio has positive net return and a daily
event t-statistic of at least `1.65`.

## Conceptual Description

Adding `PL`, `PA`, and `ALI` to the original `GC/SI/HG` metals residual basket improves robustness by broadening the metals complex while preserving the same daily relative-value mean-reversion mechanism.

## Experiment Design

- Roots: `GC, SI, HG, PL, PA, ALI`
- Asset groups: `n/a`
- Pair scope: `n/a`
- Lookback: `126` bars
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
