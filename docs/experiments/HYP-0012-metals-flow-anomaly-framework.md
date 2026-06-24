# HYP-0012-metals-flow-anomaly-framework: Metals cross-sectional flow anomaly framework

> This page is part of Francisco Amadeo's proprietary quantitative research record. Public access is provided for review and documentation only; no license is granted to copy, reuse, redistribute, commercialize, or implement the research, strategy logic, or derived conclusions.

- Status: `research`
- Owner: `famadeo`
- Decision notes: Research harness for cross-sectional dollar bars, flow geometry, Mahalanobis anomalies, trade-size flow, and relative-value diagnostics.

## Hypothesis

Cross-sectional trading flow anomalies in `GC`, `SI`, `HG`, `PL`, `PA`, and `ALI`
identify temporary dislocations in the metals complex. The dislocations may
resolve through momentum, mean reversion, relative-value convergence, or
liquidity exhaustion.

## Test

Build synchronized cross-sectional dollar bars at `$100M`, `$250M`, `$500M`, and
`$1B` total notional thresholds. On the primary `$250M` bars, estimate flow
contribution geometry, Mahalanobis anomaly distance, concentration and
dispersion features, trade-size contribution disagreement, rolling relative-value
residuals, cointegration diagnostics, and forward returns at 1, 2, 5, 10, 20,
and 50 bars.

## Decision Standard

This experiment is a research framework. A trading claim requires a later
strategy-specific backtest with explicit execution assumptions and profits at
least three times realized transaction costs.

## Conceptual Description

Cross-sectional trading flow anomalies in `GC`, `SI`, `HG`, `PL`, `PA`, and `ALI` identify temporary dislocations in the metals complex. The dislocations may resolve through momentum, mean reversion, relative-value convergence, or liquidity exhaustion.

## Experiment Design

- Roots: `GC, SI, HG, PL, PA, ALI`
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

- Completed at: `2026-06-24T00:59:07.092933+00:00`

No publishable method-level metrics were available in this result artifact.

### Summary Highlights

| Metric | Value |
|---|---:|
| `start` | 2026-05-24T00:00:00Z |
| `end` | 2026-06-22T00:00:00Z |
| `primary_complete_bars` | 6,204 |
| `trade_rows` | 3,292,329 |
| `cointegration_pairs_p_lt_0_05` | 1 |

## Publication Notes

- Proprietary work by Francisco Amadeo. All rights reserved.
- Public access does not grant permission to copy, reuse, redistribute, commercialize, or implement this research.
- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
