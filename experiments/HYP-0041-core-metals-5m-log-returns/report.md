# Core Metals 5-Minute Log Returns

Generated from the 10-year continuous 1-minute series.

Method:

- Sample each metal to 5-minute bars using the last continuous log price in each bucket.
- Align all metals to the common observed 5-minute timestamp grid.
- Forward-fill missing price, close, active-contract, and last-observation fields within each metal.
- Compute `log_return_5m = log_price_t - log_price_{t-1}` from the forward-filled log prices. Leading returns are set to `0`.
- Missing aligned bars therefore produce zero return until the next observed price update, instead of repeating the prior return.
- The long file preserves observed-only returns in `log_return_5m_raw` and marks aligned bars whose price was forward-filled with `was_price_forward_filled`.

## Inventory

| root   |   observed_rows_5m |   valid_raw_returns | first_ts                  | last_ts                   |   median_obs_1m_per_5m |   max_observed_gap_minutes |   aligned_rows_5m |   price_forward_filled_rows |
|:-------|-------------------:|--------------------:|:--------------------------|:--------------------------|-----------------------:|---------------------------:|------------------:|----------------------------:|
| GC     |             680267 |              680266 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:55:00+00:00 |                 5.0000 |                  4580.0000 |            708026 |                       27759 |
| SI     |             684609 |              684608 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:55:00+00:00 |                 5.0000 |                  4580.0000 |            708026 |                       23417 |
| HG     |             692528 |              692527 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:55:00+00:00 |                 5.0000 |                  4580.0000 |            708026 |                       15498 |
| PL     |             694073 |              694072 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:55:00+00:00 |                 5.0000 |                  4590.0000 |            708026 |                       13953 |
| PA     |             514064 |              514063 | 2016-06-22 00:00:00+00:00 | 2026-06-21 23:55:00+00:00 |                 2.0000 |                  4580.0000 |            708026 |                      193962 |

## Files

- `core_metals_5m_log_returns_long.parquet`
- `core_metals_5m_log_returns_wide.parquet`
- `core_metals_5m_log_returns_wide.csv.gz`
- `data_inventory.csv`