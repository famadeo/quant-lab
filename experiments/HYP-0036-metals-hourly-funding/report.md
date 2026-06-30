# HYP-0036 Metals Hourly Funding

## Definition

Funding is modeled as annualized curve-implied carry:

`funding = ln(F_deferred / C_proxy) / T`

where `C_proxy` is the selected near/prompt futures contract and `F_deferred` is the first liquid contract at least the target number of months beyond that anchor.

- Positive funding: contango; long futures exposure pays carry.
- Negative funding: backwardation; long futures exposure earns carry.
- Units in the main CSV/report are percent annualized.

## Method

- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.
- Raw 1-minute outright futures are aggregated to hourly last marks by contract.
- Calendar-spread symbols are excluded.
- Anchor contract: highest daily-volume contract with `months_out <= 4`.
- Deferred contract: first liquid contract at least `1M`, `3M`, or `6M` beyond anchor.
- Minimum daily contract volume: 10; minimum hourly contract volume: 1.
- This is a front-futures proxy for spot/cash, not true cash funding.

## Data Inventory

| root   |   daily_rows |   hourly_rows | first_ts                  | last_ts                   |   contracts | first_date                | last_date                 |
|:-------|-------------:|--------------:|:--------------------------|:--------------------------|------------:|:--------------------------|:--------------------------|
| GC     |        22236 |        339832 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:00:00+00:00 |         120 | 2016-06-22 00:00:00+00:00 | 2026-06-21 00:00:00+00:00 |
| SI     |        16968 |        244408 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:00:00+00:00 |         120 | 2016-06-22 00:00:00+00:00 | 2026-06-21 00:00:00+00:00 |
| HG     |        22259 |        305627 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:00:00+00:00 |         120 | 2016-06-22 00:00:00+00:00 | 2026-06-21 00:00:00+00:00 |
| PL     |        10152 |        160022 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:00:00+00:00 |         114 | 2016-06-22 00:00:00+00:00 | 2026-06-21 00:00:00+00:00 |
| PA     |         6003 |         97411 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:00:00+00:00 |          68 | 2016-06-22 00:00:00+00:00 | 2026-06-21 00:00:00+00:00 |

## Summary

| root   |   target_months |   hourly_obs |   days |   median_tenor_months |   median_funding_pct_ann |   p10_funding_pct_ann |   p90_funding_pct_ann |   contango_fraction |   backwardation_fraction |   latest_funding_pct_ann |   latest_funding_z_126d | latest_anchor   | latest_far   |
|:-------|----------------:|-------------:|-------:|----------------------:|-------------------------:|----------------------:|----------------------:|--------------------:|-------------------------:|-------------------------:|------------------------:|:----------------|:-------------|
| GC     |               1 |        51812 |   3054 |                2.0000 |                   2.7923 |                0.8629 |                5.7816 |              0.9860 |                   0.0129 |                   7.6542 |                  1.7982 | GCQ6            | GCU6         |
| GC     |               3 |        45187 |   2879 |                4.0000 |                   2.7637 |                0.8372 |                5.6885 |              0.9966 |                   0.0034 |                   4.4413 |                 -0.3429 | GCQ6            | GCZ6         |
| GC     |               6 |        37034 |   2736 |                6.0000 |                   2.8271 |                0.8059 |                5.5434 |              0.9959 |                   0.0039 |                   4.1504 |                  0.5851 | GCQ6            | GCG7         |
| HG     |               1 |        32708 |   2993 |                1.0000 |                   2.3295 |               -1.0041 |                5.8986 |              0.8281 |                   0.1603 |                   6.5066 |                  0.9264 | HGN6            | HGQ6         |
| HG     |               3 |        42600 |   2844 |                3.0000 |                   2.0731 |               -0.5139 |                5.0466 |              0.8504 |                   0.1457 |                   6.1227 |                  0.0017 | HGN6            | HGZ6         |
| HG     |               6 |        29798 |   2577 |                7.0000 |                   1.9846 |               -0.3797 |                4.9896 |              0.8528 |                   0.1443 |                   5.4395 |                 -0.2037 | HGN6            | HGH7         |
| PA     |               1 |        28074 |   2269 |                3.0000 |                   0.7562 |               -2.7546 |                4.9288 |              0.5876 |                   0.4062 |                   4.8733 |                  0.1719 | PAU6            | PAZ6         |
| PA     |               3 |        28645 |   2269 |                3.0000 |                   0.7985 |               -2.7261 |                4.9966 |              0.5941 |                   0.3998 |                   4.8733 |                  0.1494 | PAU6            | PAZ6         |
| PA     |               6 |          810 |    174 |                6.0000 |                  -0.7481 |               -3.2093 |                5.2873 |              0.4481 |                   0.5519 |                   5.5049 |                  1.2765 | PAU6            | PAH7         |
| PL     |               1 |        46846 |   2844 |                3.0000 |                   2.1465 |                0.3765 |                4.2564 |              0.9334 |                   0.0636 |                   4.1714 |                  0.4332 | PLN6            | PLV6         |
| PL     |               3 |        49763 |   2844 |                3.0000 |                   2.1561 |                0.4075 |                4.3115 |              0.9364 |                   0.0605 |                   4.1714 |                  0.9282 | PLN6            | PLV6         |
| PL     |               6 |        18374 |   1669 |                6.0000 |                   2.7114 |                0.6752 |                4.3219 |              0.9552 |                   0.0438 |                   3.7260 |                  0.5254 | PLN6            | PLF7         |
| SI     |               1 |        43038 |   2969 |                2.0000 |                   3.1967 |                0.7502 |                6.0871 |              0.9532 |                   0.0397 |                   2.6024 |                 -2.9570 | SIN6            | SIQ6         |
| SI     |               3 |        44802 |   2829 |                3.0000 |                   3.1277 |                1.0086 |                5.7126 |              0.9898 |                   0.0091 |                   3.2136 |                  0.0389 | SIN6            | SIZ6         |
| SI     |               6 |        20366 |   2255 |                7.0000 |                   3.1218 |                0.9539 |                5.4568 |              0.9908 |                   0.0084 |                   4.4313 |                  0.9392 | SIN6            | SIH7         |

## Latest Snapshot

| root   |   target_months |   hourly_obs |   days |   median_tenor_months |   median_funding_pct_ann |   p10_funding_pct_ann |   p90_funding_pct_ann |   contango_fraction |   backwardation_fraction |   latest_funding_pct_ann |   latest_funding_z_126d | latest_anchor   | latest_far   |
|:-------|----------------:|-------------:|-------:|----------------------:|-------------------------:|----------------------:|----------------------:|--------------------:|-------------------------:|-------------------------:|------------------------:|:----------------|:-------------|
| GC     |               1 |        51812 |   3054 |                2.0000 |                   2.7923 |                0.8629 |                5.7816 |              0.9860 |                   0.0129 |                   7.6542 |                  1.7982 | GCQ6            | GCU6         |
| HG     |               3 |        42600 |   2844 |                3.0000 |                   2.0731 |               -0.5139 |                5.0466 |              0.8504 |                   0.1457 |                   6.1227 |                  0.0017 | HGN6            | HGZ6         |
| PA     |               1 |        28074 |   2269 |                3.0000 |                   0.7562 |               -2.7546 |                4.9288 |              0.5876 |                   0.4062 |                   4.8733 |                  0.1719 | PAU6            | PAZ6         |
| PL     |               1 |        46846 |   2844 |                3.0000 |                   2.1465 |                0.3765 |                4.2564 |              0.9334 |                   0.0636 |                   4.1714 |                  0.4332 | PLN6            | PLV6         |
| SI     |               1 |        43038 |   2969 |                2.0000 |                   3.1967 |                0.7502 |                6.0871 |              0.9532 |                   0.0397 |                   2.6024 |                 -2.9570 | SIN6            | SIQ6         |

## Pair Selection Coverage

| root   |   target_months |   days |   median_tenor_months |   anchor_contracts |   far_contracts |
|:-------|----------------:|-------:|----------------------:|-------------------:|----------------:|
| GC     |               1 |   3054 |                  2.00 |                 60 |             111 |
| GC     |               3 |   2879 |                  4.00 |                 60 |              69 |
| GC     |               6 |   2736 |                  6.00 |                 60 |              60 |
| HG     |               1 |   2993 |                  1.00 |                 50 |             114 |
| HG     |               3 |   2844 |                  3.00 |                 50 |              79 |
| HG     |               6 |   2577 |                  7.00 |                 50 |              73 |
| PA     |               1 |   2269 |                  3.00 |                 40 |              51 |
| PA     |               3 |   2269 |                  3.00 |                 40 |              40 |
| PA     |               6 |    175 |                  6.00 |                 32 |              30 |
| PL     |               1 |   2844 |                  3.00 |                 40 |              72 |
| PL     |               3 |   2844 |                  3.00 |                 40 |              40 |
| PL     |               6 |   1669 |                  6.00 |                 40 |              40 |
| SI     |               1 |   2969 |                  2.00 |                 50 |             100 |
| SI     |               3 |   2829 |                  4.00 |                 50 |              49 |
| SI     |               6 |   2255 |                  7.00 |                 50 |              50 |

## Caveats

- Without true spot/cash prices, the estimate is front-to-deferred futures carry, not cash-to-futures carry.
- Contract month timing is approximated from month codes; this is appropriate for a first research proxy but should be replaced with exact expiry/prompt calendars.
- Daily pair selection uses same-day volume to identify liquid contracts. That is fine for curve measurement, but trading simulations should use prior-day or as-of-hour selection.
- Palladium has sparse hourly coverage; PA funding estimates should receive stricter liquidity filters before being used for alpha research.

## Files

- `hourly_funding.parquet`
- `hourly_funding.csv.gz`
- `daily_funding.csv`
- `funding_summary.csv`
- `selected_pairs.csv`
- `data_inventory.csv`
- `daily_median_hourly_funding_target1m.png`
- `daily_median_hourly_funding_target3m.png`
- `daily_median_hourly_funding_target6m.png`
- `latest_funding_snapshot_heatmap.png`