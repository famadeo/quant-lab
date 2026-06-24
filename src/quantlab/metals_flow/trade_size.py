from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.metals_flow.dollar_bars import assign_trades_to_bars

BUCKET_ORDER = ("small", "medium", "large", "very_large")


def add_trade_size_buckets(
    trades: pd.DataFrame,
    roots: tuple[str, ...],
    *,
    value_col: str = "notional",
    percentiles: tuple[float, float, float] = (0.50, 0.90, 0.99),
    mode: str = "symbol",
) -> pd.DataFrame:
    if mode not in {"symbol", "global"}:
        raise ValueError("mode must be either 'symbol' or 'global'")
    out = trades.copy()
    out["size_bucket"] = pd.Categorical(
        np.repeat("medium", len(out)),
        categories=list(BUCKET_ORDER),
        ordered=True,
    )

    if mode == "global":
        thresholds = out[value_col].quantile(percentiles).to_numpy(dtype=float)
        out["size_bucket"] = _bucketize(out[value_col], thresholds)
        return out

    for root in roots:
        mask = out["root"] == root
        if not mask.any():
            continue
        thresholds = out.loc[mask, value_col].quantile(percentiles).to_numpy(dtype=float)
        out.loc[mask, "size_bucket"] = _bucketize(out.loc[mask, value_col], thresholds)
    return out


def size_bucket_contribution_vectors(
    trades: pd.DataFrame,
    bars: pd.DataFrame,
    roots: tuple[str, ...],
    *,
    mode: str = "symbol",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    bucketed = add_trade_size_buckets(trades, roots, mode=mode)
    bucketed["bar_id"] = assign_trades_to_bars(bucketed, bars)
    bucketed = bucketed.dropna(subset=["bar_id"]).copy()
    bucketed["bar_id"] = bucketed["bar_id"].astype(int)

    grouped = (
        bucketed.groupby(["bar_id", "size_bucket", "root"], observed=True)["notional"]
        .sum()
        .rename("notional")
        .reset_index()
    )
    vectors: dict[str, pd.DataFrame] = {}
    for bucket in BUCKET_ORDER:
        pivot = (
            grouped[grouped["size_bucket"] == bucket]
            .pivot_table(index="bar_id", columns="root", values="notional", aggfunc="sum")
            .reindex(index=bars.index, columns=list(roots))
            .fillna(0.0)
        )
        denominator = pivot.sum(axis=1).replace(0.0, np.nan)
        shares = pivot.div(denominator, axis=0).fillna(0.0)
        shares.index = bars.index
        vectors[bucket] = shares

    disagreement = vectors["large"].sub(vectors["small"], fill_value=0.0)
    disagreement.columns = [f"{root}_large_minus_small_share" for root in disagreement.columns]
    disagreement["large_small_l1_distance"] = (
        vectors["large"].sub(vectors["small"], fill_value=0.0).abs().sum(axis=1)
    )
    disagreement["very_large_small_l1_distance"] = (
        vectors["very_large"].sub(vectors["small"], fill_value=0.0).abs().sum(axis=1)
    )
    return vectors, disagreement


def trade_size_summary(trades: pd.DataFrame, roots: tuple[str, ...]) -> pd.DataFrame:
    bucketed = add_trade_size_buckets(trades, roots)
    return (
        bucketed.groupby(["root", "size_bucket"], observed=True)
        .agg(
            trades=("notional", "size"),
            notional=("notional", "sum"),
            median_notional=("notional", "median"),
            mean_notional=("notional", "mean"),
        )
        .reset_index()
    )


def _bucketize(values: pd.Series, thresholds: np.ndarray) -> pd.Categorical:
    conditions = [
        values <= thresholds[0],
        (values > thresholds[0]) & (values <= thresholds[1]),
        (values > thresholds[1]) & (values <= thresholds[2]),
        values > thresholds[2],
    ]
    labels = np.select(conditions, BUCKET_ORDER, default="medium")
    return pd.Categorical(labels, categories=list(BUCKET_ORDER), ordered=True)
