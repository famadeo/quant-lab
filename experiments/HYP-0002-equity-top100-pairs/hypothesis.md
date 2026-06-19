# HYP-0002: Top-100 US Equity Pairs

## Hypothesis

Pairs selected from the largest US equities by Databento-derived market capitalization may offer more idiosyncratic relative-value opportunities than the current macro futures universe, where common macro factors dominate and pair spreads show limited net edge after costs.

## Universe

- Strict futures-only baseline: locally available equity-index futures are limited to `ES`, `NQ`, `YM`, `RTY`, and `NIY`.
- Top-100 market-cap branch: US equities ranked using Databento Security Master shares outstanding and Databento equity daily closes.
- Market cap definition: `latest close * shares_outstanding`.

## Method Notes

- Do not describe top-100 market-cap equities as futures contracts.
- Build the universe point-in-time as of a fixed date before downloading intraday bars.
- Keep the broad pull explicit because Databento reference and equity bar requests may be billable.

## First Command

```bash
uv run quantlab build-equity-universe \
  --env-file /home/famadeo/research/databento-asset-browser/.env \
  --output-path data/bronze/databento/top100_us_equities.csv \
  --as-of 2026-06-18 \
  --allow-billable
```

## Strict Futures Baseline Result

Runnable config: `experiments/HYP-0002-equity-top100-pairs/config.yaml`.

- Candidate equity-index futures pairs: 10.
- Selected pairs: 2 (`ES-NIY`, `NQ-NIY`).
- Z-score portfolio: `+2.20%` total return, Sharpe `7.73`, turnover `39.82`, trades `66`.
- Mahalanobis portfolio: `-2.71%` total return, Sharpe `-5.07`, turnover `63.89`, trades `89`.
- Decomposition:
  - Z-score gross sum `+5.56%`, cost sum `1.19%`, net sum `+4.37%` across selected pair rows.
  - Mahalanobis gross sum `-3.51%`, cost sum `1.92%`, net sum `-5.42%` across selected pair rows.

Interpretation: this is an encouraging but very narrow one-month result. It is not a trading claim; the effect is concentrated in Japan-vs-US equity-index pairs and needs a longer sample.
