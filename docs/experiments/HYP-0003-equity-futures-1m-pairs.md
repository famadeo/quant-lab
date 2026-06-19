# HYP-0003-equity-futures-1m-pairs: GLBX 1-minute equity futures pairs

- Status: `revise`
- Owner: `famadeo`
- Decision notes: Broad GLBX equity futures screen. No trading claim.

## Hypothesis

A broader GLBX equity-index futures universe may expose more relative-value structure than the five-root equity-index futures baseline, especially within economically related index and sector groups.

## Experiment Design

- Roots: `ES, NQ, YM, RTY, EMD, MES, MNQ, MYM, M2K, NIY, NKD, MNI, XAE, XAF, XAK, XAP, XAV, XAY, XAB, XAU, XAI, XAR, XAZ`
- Asset groups: MajorIndex (5), MicroIndex (4), JapanIndex (3), SectorIndex (11)
- Pair scope: `intra_asset_class`
- Lookback: `390` bars
- Signal lag: `1` bars
- Rebalance interval: `30` bars
- Selection enabled: `True`
- Train fraction: `0.6`
- Fee bps: `0.5`
- Slippage bps: `1.0`

## Results

- Completed at: `2026-06-19T12:51:01.704307+00:00`
- Candidate pairs: `73`
- Selected pairs: `4`

| Method | Total Return | Sharpe | Max Drawdown | Active Fraction | Turnover | Trades |
|---|---:|---:|---:|---:|---:|---:|
| Mahalanobis | -1.61% | -0.87 | -4.20% | 41.12% | 346.85 | 630 |
| Z-score | -3.48% | -2.86 | -4.09% | 46.25% | 191.47 | 307 |
| Z-score + Mahalanobis filter | 0.15% | 0.26 | -1.82% | 16.25% | 57.68 | 104 |

### Selection Reasons

| Key | Value |
|---|---:|
| half_life_too_slow | 8 |
| insufficient_train_observations | 35 |
| low_return_correlation | 6 |
| selected | 4 |
| spread_not_stationary | 20 |

### Top Pairs By Sharpe

| Pair | Method | Asset Class | Total Return | Sharpe | Trades |
|---|---|---|---:|---:|---:|
| XAP-XAB | Mahalanobis | SectorIndex | 6.94% | 16.20 | 69 |
| XAF-XAU | Z-score + Mahalanobis filter | SectorIndex | 2.07% | 12.58 | 8 |
| XAP-XAB | Z-score | SectorIndex | 3.30% | 10.30 | 28 |
| XAP-XAB | Z-score + Mahalanobis filter | SectorIndex | -0.06% | -0.38 | 7 |
| NIY-MNI | Mahalanobis | JapanIndex | -0.74% | -0.55 | 166 |
| NIY-MNI | Z-score + Mahalanobis filter | JapanIndex | -0.27% | -0.64 | 30 |
| YM-EMD | Z-score + Mahalanobis filter | MajorIndex | -0.34% | -0.67 | 59 |
| YM-EMD | Mahalanobis | MajorIndex | -0.94% | -0.89 | 331 |
| XAF-XAU | Z-score | SectorIndex | -0.59% | -1.56 | 27 |
| YM-EMD | Z-score | MajorIndex | -1.96% | -2.41 | 162 |

## Publication Notes

- Local data roots and artifact paths are intentionally omitted.
- Raw data, parquet outputs, MLflow state, and credentials are not published.
- This page is a summary; the experiment folder remains the source of truth.
