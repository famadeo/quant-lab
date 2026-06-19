# HYP-0004: Longer-Window XAP-XAB Pairs Robustness Test

## Hypothesis

The positive `XAP-XAB` result from the short 2026 GLBX equity-futures run may be
a local sector-rotation effect. A longer pair-specific sample should reveal
whether the z-score and Mahalanobis rules continue to generate enough gross
return to clear costs.

## Data

- Source dataset: Databento `GLBX.MDP3`.
- Raw schema: `ohlcv-1m` parent-symbol futures bars.
- Raw cache:
  `/home/famadeo/research/databento-asset-browser/data/equity_futures_1m_xap_xab_2024_2026`.
- Prepared input:
  `data/silver/equity_futures_1m_xap_xab_2024_2026_continuous`.
- Download request window: `2024-01-01T00:00:00Z` to `2026-06-19T00:00:00Z`.
- Continuous `XAP` window: `2024-01-02 14:30:00+00:00` to
  `2026-06-18 19:59:00+00:00`.
- Continuous `XAB` window: `2024-01-02 14:31:00+00:00` to
  `2026-06-18 20:00:00+00:00`.

Databento warned that a few 2025 days in the request had degraded data quality.
Treat this as a robustness caveat and inspect those dates before any trading
claim.

## Method

The experiment uses the same pair-trading settings as `HYP-0003`:

- 390-bar lookback.
- One-bar execution lag.
- 30-minute rebalance.
- 30-minute minimum hold.
- 30-minute cooldown.
- 0.5 bps fee plus 1.0 bps slippage per unit of turnover.

The pair must still pass the same train-window selection filters before the
out-of-sample test segment is evaluated.

## Status

Initial run completed on 2026-06-19.

## Strict Selection Result

The pair did **not** pass the longer-window selection gate:

| Pair | Selected | Reason | Observations | Train | Test | Return corr | Price corr | ADF p-value | Half-life bars |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `XAP-XAB` | false | `spread_not_stationary` | 31,885 | 19,131 | 12,754 | 0.347 | 0.254 | 0.197 | 1513.4 |

The longer training window runs from `2024-01-02 14:31:00+00:00` to
`2025-10-29 18:36:00+00:00`. The strict ADF threshold is `0.10`, so an ADF
p-value of `0.197` is a clear failure. This weakens the original short-window
claim substantially.

## Forced Backtest Result

Because the pair failed strict selection, a separate forced config
(`config_forced.yaml`) was run to quantify what the same rules would have done
on the longer out-of-sample segment. This is diagnostic only.

Forced out-of-sample window:

- `2025-10-29 18:46:00+00:00` to `2026-06-18 19:59:00+00:00`.
- 12,754 aligned one-minute observations.

| Method | Gross return | Cost sum | Net return | Sharpe | Max drawdown | Active fraction | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z-score | 7.93% | 1.02% | 6.83% | 7.14 | -4.78% | 44.22% | 67.98 | 85 |
| Z-score + Mahalanobis filter | 3.79% | 0.44% | 3.33% | 6.02 | -2.43% | 16.70% | 29.57 | 35 |
| Mahalanobis | -2.42% | 2.69% | -5.01% | -3.95 | -7.28% | 63.07% | 179.23 | 205 |

The longer forced result reverses the earlier conclusion about methods:
z-score remains profitable, while Mahalanobis becomes unprofitable. The hybrid
does what it was intended to do mechanically: it reduces turnover and drawdown.
However, it also rejects profitable z-score entries, so it earns less than plain
z-score.

## Window Split

Splitting the forced out-of-sample segment around the old short-window test
start (`2026-04-08 14:02:00+00:00`):

| Method | Segment | Gross return | Cost sum | Net return | Trades |
|---|---|---:|---:|---:|---:|
| Z-score | 2025-10-29 to 2026-04-08 | 5.34% | 0.76% | 4.55% | 61 |
| Z-score | 2026-04-08 to 2026-06-18 | 2.45% | 0.26% | 2.18% | 24 |
| Z-score + Mahalanobis filter | 2025-10-29 to 2026-04-08 | 2.83% | 0.38% | 2.44% | 30 |
| Z-score + Mahalanobis filter | 2026-04-08 to 2026-06-18 | 0.94% | 0.07% | 0.87% | 5 |
| Mahalanobis | 2025-10-29 to 2026-04-08 | -1.87% | 1.72% | -3.54% | 122 |
| Mahalanobis | 2026-04-08 to 2026-06-18 | -0.57% | 0.97% | -1.53% | 83 |

These overlap results are not identical to `HYP-0003` because the strategy state
and rolling estimates are initialized from a much longer history.

## Decision

Revise. The pair is not robust under the strict longer-window stationarity gate.
The forced backtest suggests the scalar z-score rule is more robust than both
standalone Mahalanobis and the hybrid filter on this longer history. The hybrid
is useful as a risk-reduction variant, but this is diagnostic after a selection
failure and should not be treated as a trading claim.
