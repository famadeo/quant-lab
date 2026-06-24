# ruff: noqa: PLR2004
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def future_returns(log_returns: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    columns: dict[tuple[str, int], pd.Series] = {}
    for root in log_returns.columns:
        for horizon in horizons:
            future = sum(log_returns[root].shift(-step) for step in range(1, horizon + 1))
            columns[(root, horizon)] = future
    frame = pd.DataFrame(columns, index=log_returns.index)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns, names=["root", "horizon"])
    return frame


def decile_forward_study(
    feature: pd.Series,
    future: pd.DataFrame,
    *,
    feature_name: str,
    bins: int = 10,
) -> pd.DataFrame:
    feature = feature.replace([np.inf, -np.inf], np.nan)
    ranks = feature.rank(method="first")
    deciles = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
    rows = []
    for (root, horizon), returns in future.items():
        frame = pd.DataFrame({"decile": deciles, "future_return": returns}).dropna()
        if frame.empty:
            continue
        for decile, group in frame.groupby("decile"):
            stats = _return_stats(group["future_return"])
            rows.append(
                {
                    "feature": feature_name,
                    "root": root,
                    "horizon": int(horizon),
                    "bucket": int(decile) + 1,
                    "bucket_type": "decile",
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def threshold_forward_study(
    mask: pd.Series,
    future: pd.DataFrame,
    *,
    feature_name: str,
    bucket_name: str,
) -> pd.DataFrame:
    mask = mask.fillna(False).astype(bool)
    rows = []
    for (root, horizon), returns in future.items():
        selected = returns[mask].dropna()
        baseline = returns[~mask].dropna()
        selected_stats = _return_stats(selected)
        baseline_stats = _return_stats(baseline)
        rows.append(
            {
                "feature": feature_name,
                "root": root,
                "horizon": int(horizon),
                "bucket": bucket_name,
                "bucket_type": "threshold",
                **selected_stats,
                "baseline_mean_bps": baseline_stats["mean_bps"],
                "excess_mean_bps": selected_stats["mean_bps"] - baseline_stats["mean_bps"],
            }
        )
    return pd.DataFrame(rows)


def information_coefficients(features: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_name, feature_values in features.items():
        clean_feature = feature_values.replace([np.inf, -np.inf], np.nan)
        if clean_feature.nunique(dropna=True) < 5:
            continue
        for (root, horizon), returns in future.items():
            frame = pd.DataFrame({"feature": clean_feature, "future": returns}).dropna()
            if len(frame) < 30:
                continue
            corr, pvalue = spearmanr(frame["feature"], frame["future"])
            rows.append(
                {
                    "feature": feature_name,
                    "root": root,
                    "horizon": int(horizon),
                    "spearman_ic": float(corr) if np.isfinite(corr) else np.nan,
                    "pvalue": float(pvalue) if np.isfinite(pvalue) else np.nan,
                    "observations": len(frame),
                }
            )
    return pd.DataFrame(rows)


def signal_classification(signals: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal_name, signal_values in signals.items():
        clean_signal = signal_values.replace([np.inf, -np.inf], np.nan)
        if clean_signal.nunique(dropna=True) < 5:
            continue
        for (root, horizon), returns in future.items():
            frame = pd.DataFrame({"signal": clean_signal, "future": returns}).dropna()
            if len(frame) < 30 or frame["signal"].std(ddof=1) == 0:
                continue
            fit = _slope_tstat(frame["signal"], frame["future"])
            sign_agreement = np.sign(frame["signal"]) == np.sign(frame["future"])
            mechanism = "momentum" if fit["beta"] > 0 else "mean_reversion"
            rows.append(
                {
                    "signal": signal_name,
                    "root": root,
                    "horizon": int(horizon),
                    "mechanism": mechanism,
                    "beta": fit["beta"],
                    "tstat": fit["tstat"],
                    "correlation": fit["correlation"],
                    "sign_agreement": float(sign_agreement.mean()),
                    "observations": len(frame),
                }
            )
    return pd.DataFrame(rows)


def event_study_paths(
    returns: pd.DataFrame,
    event_mask: pd.Series,
    *,
    window_before: int = 20,
    window_after: int = 50,
) -> pd.DataFrame:
    mask = event_mask.fillna(False).astype(bool)
    event_positions = np.flatnonzero(mask.to_numpy())
    rows = []
    for event_number, position in enumerate(event_positions):
        start = max(0, position - window_before)
        end = min(len(returns), position + window_after + 1)
        rel_index = np.arange(start - position, end - position)
        for root in returns.columns:
            values = []
            for rel in rel_index:
                if rel == 0:
                    values.append(0.0)
                elif rel > 0:
                    values.append(
                        float(returns[root].iloc[position + 1 : position + rel + 1].sum())
                    )
                else:
                    values.append(
                        float(-returns[root].iloc[position + rel + 1 : position + 1].sum())
                    )
            for rel, value in zip(rel_index, values, strict=True):
                rows.append(
                    {
                        "event_number": event_number,
                        "relative_bar": int(rel),
                        "root": root,
                        "cum_log_return": float(value),
                    }
                )
    return pd.DataFrame(rows)


def summarize_event_study(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return paths
    return (
        paths.groupby(["root", "relative_bar"], as_index=False)["cum_log_return"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )


def _return_stats(values: pd.Series) -> dict[str, float | int]:
    values = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    observations = len(values)
    if observations == 0:
        return {
            "observations": 0,
            "mean_bps": np.nan,
            "median_bps": np.nan,
            "vol_bps": np.nan,
            "tstat": np.nan,
            "hit_rate": np.nan,
            "sharpe_per_event": np.nan,
        }
    std = float(values.std(ddof=1)) if observations > 1 else 0.0
    mean = float(values.mean())
    return {
        "observations": observations,
        "mean_bps": mean * 10_000.0,
        "median_bps": float(values.median()) * 10_000.0,
        "vol_bps": std * 10_000.0,
        "tstat": mean / (std / math.sqrt(observations)) if std > 0 else np.nan,
        "hit_rate": float((values > 0).mean()),
        "sharpe_per_event": mean / std if std > 0 else np.nan,
    }


def _slope_tstat(x_values: pd.Series, y_values: pd.Series) -> dict[str, float]:
    x = x_values.to_numpy(dtype=float)
    y = y_values.to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    dof = len(x) - 2
    if dof <= 0:
        return {"beta": float(beta[1]), "tstat": np.nan, "correlation": np.nan}
    sigma_sq = float(residuals @ residuals / dof)
    cov = sigma_sq * np.linalg.pinv(design.T @ design)
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
    return {
        "beta": float(beta[1]),
        "tstat": float(beta[1] / se) if se > 0 else np.nan,
        "correlation": corr,
    }
