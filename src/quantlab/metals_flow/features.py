from __future__ import annotations

import numpy as np
import pandas as pd


def contribution_matrix(
    bars: pd.DataFrame,
    roots: tuple[str, ...],
    *,
    value_suffix: str = "notional",
) -> pd.DataFrame:
    columns = [f"{root}_{value_suffix}" for root in roots]
    values = bars.loc[:, columns].astype(float).copy()
    values.columns = list(roots)
    denominator = values.sum(axis=1).replace(0.0, np.nan)
    shares = values.div(denominator, axis=0).fillna(0.0)
    shares.index = bars.index
    return shares


def signed_flow_matrix(bars: pd.DataFrame, roots: tuple[str, ...]) -> pd.DataFrame:
    signed = bars.loc[:, [f"{root}_signed_notional" for root in roots]].astype(float)
    signed.columns = list(roots)
    denominator = bars["bar_notional"].astype(float).replace(0.0, np.nan)
    return signed.div(denominator, axis=0).fillna(0.0)


def concentration_features(shares: pd.DataFrame) -> pd.DataFrame:
    values = shares.clip(lower=0.0).to_numpy(dtype=float)
    n_assets = values.shape[1]
    safe = np.where(values > 0.0, values, np.nan)
    entropy = -np.nansum(safe * np.log(safe), axis=1)
    hhi = np.square(values).sum(axis=1)
    sorted_values = np.sort(values, axis=1)
    index = np.arange(1, n_assets + 1, dtype=float)
    row_sums = sorted_values.sum(axis=1)
    gini = np.where(
        row_sums > 0.0,
        (2.0 * (sorted_values * index).sum(axis=1)) / (n_assets * row_sums)
        - (n_assets + 1.0) / n_assets,
        np.nan,
    )
    equal_weight = np.full(n_assets, 1.0 / n_assets)
    max_share = shares.max(axis=1)
    dominant_root = shares.idxmax(axis=1)

    return pd.DataFrame(
        {
            "entropy": entropy,
            "entropy_normalized": entropy / np.log(n_assets),
            "hhi": hhi,
            "effective_metals": np.exp(entropy),
            "gini": gini,
            "share_variance": shares.var(axis=1, ddof=0),
            "distance_from_equal_weight": np.sqrt(np.square(values - equal_weight).sum(axis=1)),
            "max_share": max_share,
            "dominant_root": dominant_root,
        },
        index=shares.index,
    )


def dynamic_features(shares: pd.DataFrame) -> pd.DataFrame:
    velocity = shares.diff().fillna(0.0)
    acceleration = velocity.diff().fillna(0.0)
    ranks = shares.rank(axis=1, ascending=False, method="min")
    rank_change = ranks.diff().fillna(0.0)
    out = pd.DataFrame(
        {
            "contribution_velocity_l2": np.sqrt(np.square(velocity).sum(axis=1)),
            "contribution_acceleration_l2": np.sqrt(np.square(acceleration).sum(axis=1)),
            "rank_turnover_l1": rank_change.abs().sum(axis=1),
        },
        index=shares.index,
    )
    for root in shares.columns:
        out[f"{root}_share_velocity"] = velocity[root]
        out[f"{root}_share_acceleration"] = acceleration[root]
        out[f"{root}_share_rank"] = ranks[root]
        out[f"{root}_share_rank_change"] = rank_change[root]
    return out


def rolling_zscores(
    frame: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    min_periods = min_periods or max(10, window // 5)
    mean = frame.rolling(window=window, min_periods=min_periods).mean()
    std = frame.rolling(window=window, min_periods=min_periods).std(ddof=1)
    return ((frame - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def assemble_flow_features(
    bars: pd.DataFrame,
    roots: tuple[str, ...],
    *,
    rolling_window: int = 500,
    min_periods: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shares = contribution_matrix(bars, roots)
    signed = signed_flow_matrix(bars, roots)
    features = pd.concat(
        [
            concentration_features(shares),
            dynamic_features(shares),
        ],
        axis=1,
    )
    features["complex_signed_notional_ratio"] = bars["complex_signed_notional_ratio"].astype(float)
    features["abs_complex_signed_notional_ratio"] = features["complex_signed_notional_ratio"].abs()
    for root in roots:
        features[f"{root}_notional_share"] = shares[root]
        features[f"{root}_signed_notional_share"] = signed[root]
        features[f"{root}_abs_signed_notional_share"] = signed[root].abs()

    z_input = pd.concat(
        [
            shares.add_suffix("_share"),
            signed.add_suffix("_signed_share"),
            features[
                [
                    "entropy",
                    "hhi",
                    "effective_metals",
                    "distance_from_equal_weight",
                    "contribution_velocity_l2",
                    "abs_complex_signed_notional_ratio",
                ]
            ],
        ],
        axis=1,
    )
    zscores = rolling_zscores(z_input, rolling_window, min_periods).add_suffix("_z")
    return features, shares, zscores
