# HYP-0003: GLBX 1-Minute Equity Futures Pairs

## Hypothesis

A broader GLBX equity-index futures universe may expose more relative-value
structure than the five-root equity-index futures baseline, especially within
economically related index and sector groups.

## Data

- Source dataset: Databento `GLBX.MDP3`.
- Raw schema: `ohlcv-1m` root-level outright futures bars.
- Raw cache: `/home/famadeo/research/databento-asset-browser/data/equity_futures_1m`.
- Prepared input: `data/silver/equity_futures_1m_continuous`.
- Prepared roots: `ES`, `NQ`, `YM`, `RTY`, `EMD`, `MES`, `MNQ`, `MYM`, `M2K`, `NIY`, `NKD`, `MNI`, `XAE`, `XAF`, `XAK`, `XAP`, `XAV`, `XAY`, `XAB`, `XAU`, `XAI`, `XAR`, `XAZ`.

## Continuous-Series Construction

Each root is converted from outright contracts into a return-spliced continuous
series. The active contract is selected daily using forward-only volume
dominance: a later-expiring contract can become active only after its daily
volume exceeds the current active contract. The continuous price is reconstructed
from active-contract log returns, avoiding a raw price jump at roll dates.

## Method

The experiment compares the existing z-score pairs strategy against the
Mahalanobis-distance variant on the same selected pairs. Candidate pairs are
restricted to intra-group combinations:

- Major index futures.
- Micro index futures.
- Japan index futures.
- Sector index futures.

Signals use a 390-bar lookback, a one-bar execution lag, 30-minute rebalance,
30-minute minimum hold, and 30-minute cooldown. Costs are set to 0.5 bps fee and
1.0 bps slippage per unit of turnover, matching the previous pairs experiments.

## Status

Initial run completed on 2026-06-19.

## Result

- Evaluated pairs after minimum-observation filtering: 73.
- Selected pairs: 4 (`YM-EMD`, `NIY-MNI`, `XAF-XAU`, `XAP-XAB`).
- Selection rejects:
  - 35 insufficient train observations.
  - 20 non-stationary spreads.
  - 8 half-life too slow.
  - 6 low return correlation.

Portfolio-level results:

| Method | Total return | Sharpe | Max drawdown | Active fraction | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Mahalanobis | -1.61% | -0.87 | -4.20% | 41.12% | 346.85 | 630 |
| Z-score | -3.48% | -2.86 | -4.09% | 46.25% | 191.47 | 307 |
| Z-score + Mahalanobis filter | 0.15% | 0.26 | -1.82% | 16.25% | 57.68 | 104 |

Gross/cost decomposition using the portfolio aggregation convention:

| Method | Gross compounded | Cost sum | Net compounded |
|---|---:|---:|---:|
| Mahalanobis | 3.64% | 5.20% | -1.61% |
| Z-score | -0.67% | 2.87% | -3.48% |
| Z-score + Mahalanobis filter | 1.02% | 0.87% | 0.15% |

The hybrid method uses z-score for direction and exits, but only enters or
reverses when the rolling Mahalanobis distance is above the same outlier
threshold used by the standalone Mahalanobis method.

For `XAP-XAB`, the hybrid reduced turnover sharply but filtered out too much of
the profitable short-window move:

| Method | Gross return | Cost sum | Net return | Trades |
|---|---:|---:|---:|---:|
| Mahalanobis | 7.92% | 0.91% | 6.94% | 69 |
| Z-score | 3.64% | 0.32% | 3.30% | 28 |
| Z-score + Mahalanobis filter | 0.04% | 0.10% | -0.06% | 7 |

The broader result is still not a trading claim: selected-pair count is small,
edge is concentrated, and the best-performing filter depends on the date window.

## Decision

Revise. The broader GLBX equity futures universe is useful for research, but the
current rule set only barely clears costs under the hybrid portfolio result. The
next iteration should focus on contract-specific cost assumptions, robustness
checks around `XAP-XAB`, and parameter sensitivity for the Mahalanobis entry
filter rather than scaling capital.
