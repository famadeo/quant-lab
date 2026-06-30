# HYP-0019-metals-flow-residual-state-momentum: Residual-state momentum filtered by extreme metals flow geometry

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `reject`
- Owner: `famadeo`
- Decision notes: Direct PA/PL residual-state momentum lost money after costs; q99 flow filtering reduced turnover but not gross losses, and the 3x-cost gate admitted no roots.

## Hypothesis

When the metals complex enters an extreme cross-sectional flow-geometry regime, fair-value residual dislocations in the less liquid PGM roots do not immediately converge. Instead, residual states in `PL` and `PA` persist over the next 10-20 cross-sectional dollar bars.

## Economic Mechanism

The expected edge is slow risk transfer. `GC`, `SI`, and `HG` dominate flow and absorb information faster. `PL` and especially `PA` are thinner, so extreme complex-wide flow states can coincide with delayed inventory transfer, wider liquidity provision, and continued repricing before convergence.

## Test

Use the corrected three-year core framework from HYP-0018:

- all-outright trade flow;
- roll-adjusted continuous 1-minute marks for returns;
- shifted walk-forward Mahalanobis thresholds;
- MBP1-derived spread cost estimates;
- train/test split with embargo.

The strategy enters in the direction of the residual z-score and exits on residual decay, sign reversal, maximum holding period, invalid price marks, or stop z-score.

## Success Criteria

A candidate variant should satisfy:

- positive net return after costs;
- gross return at least 3x estimated transaction costs;
- positive purged-test performance;
- economically plausible trade count and holding period.

## Conceptual Description

When the metals complex enters an extreme cross-sectional flow-geometry regime, fair-value residual dislocations in the less liquid PGM roots do not immediately converge. Instead, residual states in `PL` and `PA` persist over the next 10-20 cross-sectional dollar bars.

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
