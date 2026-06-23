# HYP-0008-cross-asset-residual-baskets: Cross-asset residual mean-reversion baskets

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Positive OOS net return but below the preregistered t-stat threshold.

## Hypothesis

Within related futures groups, normalized price residuals mean-revert. A daily
close signal that fades residual z-scores should produce positive net return out
of sample.

## Method

Use daily roll-adjusted continuous futures from 2010-2024. For each asset group,
compute each root's residual log price versus the group mean, convert it to a
rolling z-score, and fade it with one-day execution lag. Positions are normalized
to one unit of gross exposure.

## Decision Rule

Reject unless the out-of-sample portfolio net return is positive and the daily
event t-statistic is at least 1.65 after turnover costs.

## Conceptual Description

Within related futures groups, normalized price residuals mean-revert. A daily close signal that fades residual z-scores should produce positive net return out of sample.

## Experiment Design

- Roots: `6A, 6B, 6C, 6E, 6J, ES, NQ, GC, HG, SI, ZT, ZF, ZN, ZB`
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
