# ruff: noqa: PLR2004, PERF401
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.metals_flow.anomaly import build_anomaly_frame
from quantlab.metals_flow.config import MetalsFlowConfig
from quantlab.metals_flow.dollar_bars import (
    bar_log_returns,
    endpoint_prices,
    load_or_build_bars,
    load_or_build_trade_cache,
    summarize_bars,
)
from quantlab.metals_flow.fair_value import (
    pairwise_cointegration,
    rolling_relative_value_residuals,
)
from quantlab.metals_flow.features import assemble_flow_features
from quantlab.metals_flow.forward import (
    decile_forward_study,
    event_study_paths,
    future_returns,
    information_coefficients,
    signal_classification,
    summarize_event_study,
    threshold_forward_study,
)
from quantlab.metals_flow.geometry import run_geometry_suite
from quantlab.metals_flow.orderbook import mbp1_availability
from quantlab.metals_flow.plots import (
    plot_concentration,
    plot_event_study,
    plot_fair_value_zscores,
    plot_forward_heatmap,
    plot_geometry,
    plot_mahalanobis,
    plot_rolling_contribution,
    plot_threshold_sensitivity,
    plot_trade_size_disagreement,
)
from quantlab.metals_flow.trade_size import (
    size_bucket_contribution_vectors,
    trade_size_summary,
)


@dataclass(frozen=True)
class MetalsFlowResearchResult:
    results_path: Path
    artifacts: dict[str, str]
    summary: dict[str, Any]


def run_metals_flow_research(config: MetalsFlowConfig) -> MetalsFlowResearchResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = config.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    trades = load_or_build_trade_cache(config)
    bars_by_threshold = load_or_build_bars(config, trades)
    threshold_summary = summarize_bars(bars_by_threshold)
    threshold_summary_path = config.output_dir / "threshold_summary.csv"
    threshold_summary.to_csv(threshold_summary_path, index=False)

    primary_bars = (
        bars_by_threshold[float(config.primary_threshold)]
        .loc[lambda frame: frame["complete"]]
        .reset_index(drop=True)
    )
    prices = endpoint_prices(trades, primary_bars, config.roots).reset_index(drop=True)
    log_prices = np.log(prices.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).ffill()
    returns = bar_log_returns(prices).reset_index(drop=True)

    features, shares, zscores = assemble_flow_features(
        primary_bars,
        config.roots,
        rolling_window=config.rolling_window,
        min_periods=config.min_periods,
    )
    anomalies = build_anomaly_frame(
        shares,
        window=config.rolling_window,
        min_periods=config.min_periods,
        ewma_halflife=config.ewma_halflife,
    )
    future = future_returns(returns, config.horizons)

    size_summary = trade_size_summary(trades, config.roots)
    size_vectors, size_disagreement = size_bucket_contribution_vectors(
        trades, primary_bars, config.roots
    )

    fair_value_residuals, fair_value_zscores = rolling_relative_value_residuals(
        log_prices,
        lookback=config.fair_value_lookback,
        min_periods=config.fair_value_min_periods,
    )
    cointegration = pairwise_cointegration(log_prices, min_obs=config.fair_value_min_periods)

    geometry = run_geometry_suite(shares)
    forward_studies = _build_forward_studies(features, anomalies, size_disagreement, future)
    ic = information_coefficients(
        _numeric_feature_subset(features, anomalies, size_disagreement),
        future,
    )
    classifications = signal_classification(
        _classification_signals(features, anomalies, size_disagreement, fair_value_zscores),
        future,
    )

    event_paths = event_study_paths(returns, anomalies["md_q95"], window_before=20, window_after=50)
    event_summary = summarize_event_study(event_paths)
    book_availability = mbp1_availability(config.mbp1_dir)

    artifacts = _write_artifacts(
        config=config,
        primary_bars=primary_bars,
        prices=prices,
        returns=returns,
        shares=shares,
        features=features,
        zscores=zscores,
        anomalies=anomalies,
        size_summary=size_summary,
        size_vectors=size_vectors,
        size_disagreement=size_disagreement,
        fair_value_residuals=fair_value_residuals,
        fair_value_zscores=fair_value_zscores,
        cointegration=cointegration,
        geometry=geometry,
        forward_studies=forward_studies,
        ic=ic,
        classifications=classifications,
        event_summary=event_summary,
        book_availability=book_availability,
    )
    artifacts.update(
        _write_plots(
            config=config,
            plot_dir=plot_dir,
            threshold_summary=threshold_summary,
            primary_bars=primary_bars,
            shares=shares,
            features=features,
            anomalies=anomalies,
            geometry=geometry,
            forward_studies=forward_studies,
            size_disagreement=size_disagreement,
            fair_value_zscores=fair_value_zscores,
            event_summary=event_summary,
        )
    )

    summary = _summary_payload(
        config=config,
        trades=trades,
        primary_bars=primary_bars,
        threshold_summary=threshold_summary,
        shares=shares,
        anomalies=anomalies,
        forward_studies=forward_studies,
        ic=ic,
        classifications=classifications,
        cointegration=cointegration,
        book_availability=book_availability,
        geometry_skipped=geometry.skipped,
    )
    results_path = config.output_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "experiment_id": config.experiment_id,
                "title": config.title,
                "completed_at": datetime.now(UTC).isoformat(),
                "summary": summary,
                "artifacts": artifacts,
            },
            indent=2,
            default=str,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return MetalsFlowResearchResult(results_path=results_path, artifacts=artifacts, summary=summary)


def _build_forward_studies(
    features: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    studies = [
        decile_forward_study(anomalies["md_rolling"], future, feature_name="md_rolling"),
        decile_forward_study(
            features["entropy_normalized"], future, feature_name="entropy_normalized"
        ),
        decile_forward_study(features["hhi"], future, feature_name="hhi"),
        decile_forward_study(
            features["distance_from_equal_weight"],
            future,
            feature_name="distance_from_equal_weight",
        ),
        decile_forward_study(
            size_disagreement["large_small_l1_distance"],
            future,
            feature_name="large_small_l1_distance",
        ),
    ]
    for label in ("md_q90", "md_q95", "md_q99"):
        if label in anomalies:
            studies.append(
                threshold_forward_study(
                    anomalies[label],
                    future,
                    feature_name="md_rolling",
                    bucket_name=label,
                )
            )
    return pd.concat(studies, ignore_index=True)


def _numeric_feature_subset(
    features: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        features.select_dtypes(include=[np.number]),
        anomalies[["md_rolling", "md_expanding", "md_ewma", "md_robust_snapshot"]],
        size_disagreement,
    ]
    return pd.concat(columns, axis=1).replace([np.inf, -np.inf], np.nan)


def _classification_signals(
    features: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    fair_value_zscores: pd.DataFrame,
) -> pd.DataFrame:
    signal_parts = [
        features.filter(regex="signed_notional_share$"),
        features[["complex_signed_notional_ratio"]],
        anomalies[["md_rolling"]],
        size_disagreement.filter(regex="large_minus_small_share$"),
        fair_value_zscores.add_prefix("rv_z_"),
    ]
    return pd.concat(signal_parts, axis=1).replace([np.inf, -np.inf], np.nan)


def _write_artifacts(
    *,
    config: MetalsFlowConfig,
    primary_bars: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    shares: pd.DataFrame,
    features: pd.DataFrame,
    zscores: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_summary: pd.DataFrame,
    size_vectors: dict[str, pd.DataFrame],
    size_disagreement: pd.DataFrame,
    fair_value_residuals: pd.DataFrame,
    fair_value_zscores: pd.DataFrame,
    cointegration: pd.DataFrame,
    geometry: Any,
    forward_studies: pd.DataFrame,
    ic: pd.DataFrame,
    classifications: pd.DataFrame,
    event_summary: pd.DataFrame,
    book_availability: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "bars": config.output_dir / "primary_bars.parquet",
        "prices": config.output_dir / "bar_endpoint_prices.parquet",
        "returns": config.output_dir / "bar_returns.parquet",
        "shares": config.output_dir / "contribution_shares.parquet",
        "features": config.output_dir / "flow_features.parquet",
        "zscores": config.output_dir / "flow_feature_zscores.parquet",
        "anomalies": config.output_dir / "flow_anomalies.parquet",
        "trade_size_summary": config.output_dir / "trade_size_summary.csv",
        "trade_size_disagreement": config.output_dir / "trade_size_disagreement.parquet",
        "fair_value_residuals": config.output_dir / "fair_value_residuals.parquet",
        "fair_value_zscores": config.output_dir / "fair_value_zscores.parquet",
        "cointegration": config.output_dir / "cointegration.csv",
        "forward_studies": config.output_dir / "forward_studies.csv",
        "information_coefficients": config.output_dir / "information_coefficients.csv",
        "signal_classification": config.output_dir / "signal_classification.csv",
        "event_study_summary": config.output_dir / "event_study_summary.csv",
        "book_availability": config.output_dir / "mbp1_availability.csv",
    }
    primary_bars.to_parquet(paths["bars"], index=False)
    prices.to_parquet(paths["prices"], index=False)
    returns.to_parquet(paths["returns"], index=False)
    shares.to_parquet(paths["shares"], index=False)
    features.to_parquet(paths["features"], index=False)
    zscores.to_parquet(paths["zscores"], index=False)
    anomalies.to_parquet(paths["anomalies"], index=False)
    size_summary.to_csv(paths["trade_size_summary"], index=False)
    size_disagreement.to_parquet(paths["trade_size_disagreement"], index=False)
    fair_value_residuals.to_parquet(paths["fair_value_residuals"], index=False)
    fair_value_zscores.to_parquet(paths["fair_value_zscores"], index=False)
    cointegration.to_csv(paths["cointegration"], index=False)
    forward_studies.to_csv(paths["forward_studies"], index=False)
    ic.to_csv(paths["information_coefficients"], index=False)
    classifications.to_csv(paths["signal_classification"], index=False)
    event_summary.to_csv(paths["event_study_summary"], index=False)
    book_availability.to_csv(paths["book_availability"], index=False)

    for bucket, vector in size_vectors.items():
        path = config.output_dir / f"trade_size_{bucket}_shares.parquet"
        vector.to_parquet(path, index=False)
        paths[f"trade_size_{bucket}_shares"] = path
    for name, coords in geometry.coordinates.items():
        path = config.output_dir / f"geometry_{name}.csv"
        coords.to_csv(path)
        paths[f"geometry_{name}"] = path
    geometry.pca_loadings.to_csv(config.output_dir / "geometry_pca_loadings.csv")
    geometry.pca_explained_variance.to_csv(config.output_dir / "geometry_pca_explained.csv")
    geometry.clusters.to_csv(config.output_dir / "geometry_clusters.csv")
    paths["geometry_pca_loadings"] = config.output_dir / "geometry_pca_loadings.csv"
    paths["geometry_pca_explained"] = config.output_dir / "geometry_pca_explained.csv"
    paths["geometry_clusters"] = config.output_dir / "geometry_clusters.csv"
    return {key: str(value) for key, value in paths.items()}


def _write_plots(
    *,
    config: MetalsFlowConfig,
    plot_dir: Path,
    threshold_summary: pd.DataFrame,
    primary_bars: pd.DataFrame,
    shares: pd.DataFrame,
    features: pd.DataFrame,
    anomalies: pd.DataFrame,
    geometry: Any,
    forward_studies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    fair_value_zscores: pd.DataFrame,
    event_summary: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "plot_threshold_sensitivity": plot_dir / "threshold_sensitivity.png",
        "plot_rolling_contribution": plot_dir / "rolling_contribution.png",
        "plot_mahalanobis": plot_dir / "mahalanobis.png",
        "plot_concentration": plot_dir / "concentration.png",
        "plot_event_study": plot_dir / "event_study_q95.png",
        "plot_forward_md": plot_dir / "forward_md_rolling_h10.png",
        "plot_trade_size_disagreement": plot_dir / "trade_size_disagreement.png",
        "plot_fair_value_zscores": plot_dir / "fair_value_zscores.png",
    }
    plot_threshold_sensitivity(threshold_summary, paths["plot_threshold_sensitivity"])
    plot_rolling_contribution(shares, primary_bars, paths["plot_rolling_contribution"])
    plot_mahalanobis(anomalies, primary_bars, paths["plot_mahalanobis"])
    plot_concentration(features, primary_bars, paths["plot_concentration"])
    plot_event_study(event_summary, paths["plot_event_study"])
    plot_forward_heatmap(
        forward_studies,
        paths["plot_forward_md"],
        feature="md_rolling",
        horizon=10 if 10 in config.horizons else config.horizons[0],
    )
    plot_trade_size_disagreement(
        size_disagreement, primary_bars, paths["plot_trade_size_disagreement"]
    )
    time_indexed_rv = fair_value_zscores.copy()
    time_indexed_rv.index = pd.to_datetime(primary_bars["end_ts"], utc=True)
    plot_fair_value_zscores(time_indexed_rv, paths["plot_fair_value_zscores"])

    if "pca" in geometry.coordinates:
        pca_path = plot_dir / "geometry_pca_md.png"
        plot_geometry(
            geometry.coordinates["pca"],
            anomalies["md_rolling"],
            pca_path,
            title="PCA contribution geometry colored by rolling MD",
        )
        paths["plot_geometry_pca"] = pca_path
    if "tsne" in geometry.coordinates:
        tsne_path = plot_dir / "geometry_tsne_md.png"
        plot_geometry(
            geometry.coordinates["tsne"],
            anomalies["md_rolling"],
            tsne_path,
            title="t-SNE contribution geometry colored by rolling MD",
        )
        paths["plot_geometry_tsne"] = tsne_path

    return {key: str(value) for key, value in paths.items()}


def _summary_payload(
    *,
    config: MetalsFlowConfig,
    trades: pd.DataFrame,
    primary_bars: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    shares: pd.DataFrame,
    anomalies: pd.DataFrame,
    forward_studies: pd.DataFrame,
    ic: pd.DataFrame,
    classifications: pd.DataFrame,
    cointegration: pd.DataFrame,
    book_availability: pd.DataFrame,
    geometry_skipped: dict[str, str],
) -> dict[str, Any]:
    top_ic = (
        ic.assign(abs_ic=ic["spearman_ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .head(20)
        .drop(columns=["abs_ic"])
        .to_dict(orient="records")
        if not ic.empty
        else []
    )
    top_forward = (
        forward_studies.sort_values("mean_bps", ascending=False).head(20).to_dict(orient="records")
        if not forward_studies.empty
        else []
    )
    mechanism_counts = (
        classifications["mechanism"].value_counts().to_dict() if not classifications.empty else {}
    )
    return {
        "roots": list(config.roots),
        "start": config.start,
        "end": config.end,
        "trade_rows": len(trades),
        "primary_threshold": config.primary_threshold,
        "primary_complete_bars": len(primary_bars),
        "threshold_summary": threshold_summary.to_dict(orient="records"),
        "mean_contribution": shares.mean().to_dict(),
        "dominant_share": primary_bars["dominant_root"].value_counts(normalize=True).to_dict(),
        "md_rolling_quantiles": anomalies["md_rolling"].quantile([0.90, 0.95, 0.99]).to_dict(),
        "anomaly_counts": {
            key: int(anomalies[key].sum())
            for key in ("md_q90", "md_q95", "md_q99")
            if key in anomalies
        },
        "top_forward_buckets": top_forward,
        "top_information_coefficients": top_ic,
        "mechanism_counts": mechanism_counts,
        "cointegration_pairs_p_lt_0_05": int(
            (cointegration.get("coint_pvalue", pd.Series(dtype=float)) < 0.05).sum()
        ),
        "mbp1_availability": book_availability.to_dict(orient="records"),
        "geometry_skipped": geometry_skipped,
    }
