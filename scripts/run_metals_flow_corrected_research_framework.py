# ruff: noqa: PLR2004
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from quantlab.metals_flow.anomaly import build_anomaly_frame
from quantlab.metals_flow.config import MetalsFlowConfig
from quantlab.metals_flow.data_quality import (
    align_continuous_marks_to_bars,
    bar_log_returns_with_validity,
)
from quantlab.metals_flow.dollar_bars import summarize_bars
from quantlab.metals_flow.fair_value import (
    pairwise_cointegration,
    rolling_relative_value_residuals,
)
from quantlab.metals_flow.features import assemble_flow_features
from quantlab.metals_flow.forward import (
    event_study_paths,
    future_returns,
    information_coefficients,
    signal_classification,
    summarize_event_study,
)
from quantlab.metals_flow.geometry import run_geometry_suite
from quantlab.metals_flow.orderbook import mbp1_availability
from quantlab.metals_flow.runner import (
    _build_forward_studies,
    _classification_signals,
    _numeric_feature_subset,
    _write_plots,
)

sys.path.append(str(Path(__file__).resolve().parent))
from run_metals_flow_corrected_residual_strategy import (
    _build_sorted_trade_cache_with_symbols,
    _size_thresholds_from_calibration,
    _stream_build_bars,
    _trade_inventory_extended,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected streaming metals flow research framework."
    )
    parser.add_argument("config", type=Path, help="Experiment config YAML.")
    return parser.parse_args()


def main() -> None:
    config = MetalsFlowConfig.from_yaml(parse_args().config)
    if config.continuous_dir is None:
        raise ValueError("corrected framework requires data.continuous_dir")
    run_corrected_research_framework(config)


def run_corrected_research_framework(config: MetalsFlowConfig) -> Path:
    roots = config.roots
    out_dir = config.output_dir
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    print("building sorted symbol-preserving trade cache", flush=True)
    sorted_path = _build_sorted_trade_cache_with_symbols(config, roots)
    trade_inventory = _trade_inventory_extended(config, roots)
    size_thresholds = _size_thresholds_from_calibration(
        sorted_path,
        roots,
        start=config.start,
        calibration_days=30,
    )
    trade_size_summary = _trade_size_summary_from_thresholds(sorted_path, roots, size_thresholds)

    print("building threshold dollar bars", flush=True)
    bars_by_threshold = _load_or_build_streaming_threshold_bars(config, sorted_path, roots)
    threshold_summary = summarize_bars(bars_by_threshold)
    threshold_summary.to_csv(out_dir / "threshold_summary.csv", index=False)

    primary_threshold = float(config.primary_threshold)
    primary_bars, size_disagreement, endpoint_symbols = _load_or_build_primary_bars(
        config,
        sorted_path,
        roots,
        primary_threshold,
        size_thresholds,
    )

    print("aligning corrected continuous marks", flush=True)
    price_panel = _continuous_price_panel(config, primary_bars, roots)
    log_prices = price_panel["log_prices"].reset_index(drop=True)
    prices = np.exp(log_prices).replace([np.inf, -np.inf], np.nan)
    returns = bar_log_returns_with_validity(
        log_prices,
        price_panel["valid_price_mask"],
    ).reset_index(drop=True)
    price_validity = price_panel["price_validity"].reset_index(drop=True)
    continuous_marks = price_panel["continuous_marks"].reset_index(drop=True)

    print("computing flow features, anomalies, and forward studies", flush=True)
    features, shares, zscores = assemble_flow_features(
        primary_bars,
        roots,
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
    forward_studies = _build_forward_studies(features, anomalies, size_disagreement, future)
    ic = information_coefficients(
        _numeric_feature_subset(features, anomalies, size_disagreement),
        future,
    )

    print("computing fair-value and geometry diagnostics", flush=True)
    fair_value_residuals, fair_value_zscores = rolling_relative_value_residuals(
        log_prices,
        lookback=config.fair_value_lookback,
        min_periods=config.fair_value_min_periods,
    )
    cointegration = pairwise_cointegration(
        log_prices,
        min_obs=config.fair_value_min_periods,
        max_obs=20_000,
        maxlag=1,
        autolag=None,
    )
    classifications = signal_classification(
        _classification_signals(features, anomalies, size_disagreement, fair_value_zscores),
        future,
    )
    geometry = run_geometry_suite(shares)
    event_paths = event_study_paths(returns, anomalies["md_q95"], window_before=20, window_after=50)
    event_summary = summarize_event_study(event_paths)
    book_availability = mbp1_availability(config.mbp1_dir)

    print("writing artifacts and plots", flush=True)
    artifacts = _write_streaming_artifacts(
        config=config,
        primary_bars=primary_bars,
        prices=prices,
        returns=returns,
        shares=shares,
        features=features,
        zscores=zscores,
        anomalies=anomalies,
        size_thresholds=size_thresholds,
        size_summary=trade_size_summary,
        size_disagreement=size_disagreement,
        endpoint_symbols=endpoint_symbols,
        price_validity=price_validity,
        continuous_marks=continuous_marks,
        fair_value_residuals=fair_value_residuals,
        fair_value_zscores=fair_value_zscores,
        cointegration=cointegration,
        geometry=geometry,
        forward_studies=forward_studies,
        ic=ic,
        classifications=classifications,
        event_summary=event_summary,
        book_availability=book_availability,
        trade_inventory=trade_inventory,
        threshold_summary=threshold_summary,
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

    results_path = out_dir / "results.json"
    results = {
        "experiment_id": config.experiment_id,
        "title": config.title,
        "completed_at": datetime.now(UTC).isoformat(),
        "summary": _summary_payload(
            config=config,
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
            price_validity=price_validity,
            trade_inventory=trade_inventory,
        ),
        "artifacts": artifacts,
    }
    results_path.write_text(
        json.dumps(json_safe(results), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {results_path}", flush=True)
    return results_path


def _load_or_build_streaming_threshold_bars(
    config: MetalsFlowConfig,
    sorted_path: Path,
    roots: tuple[str, ...],
) -> dict[float, pd.DataFrame]:
    bars_by_threshold: dict[float, pd.DataFrame] = {}
    for threshold_value in config.thresholds:
        threshold = float(threshold_value)
        path = config.cache_dir / f"stream_bars_{int(threshold)}_{config.date_tag}.parquet"
        legacy_path = (
            config.cache_dir / f"corrected_bars_core_{int(threshold)}_{config.date_tag}.parquet"
        )
        if path.exists():
            bars = pd.read_parquet(path)
            bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
            bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
        elif legacy_path.exists():
            bars = pd.read_parquet(legacy_path)
            bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
            bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
        else:
            result = _stream_build_bars(
                sorted_path,
                roots,
                threshold,
                size_thresholds=None,
                include_size=False,
                include_endpoint=False,
            )
            bars = result.bars
            bars.to_parquet(path, index=False)
        bars_by_threshold[threshold] = bars
        print(f"  threshold {threshold:,.0f}: {len(bars):,} bars", flush=True)
    return bars_by_threshold


def _load_or_build_primary_bars(
    config: MetalsFlowConfig,
    sorted_path: Path,
    roots: tuple[str, ...],
    threshold: float,
    size_thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_path = config.cache_dir / f"stream_primary_{int(threshold)}_{config.date_tag}.parquet"
    size_path = (
        config.cache_dir / f"stream_size_disagreement_{int(threshold)}_{config.date_tag}.parquet"
    )
    endpoint_path = (
        config.cache_dir / f"stream_endpoint_symbols_{int(threshold)}_{config.date_tag}.parquet"
    )
    legacy_primary_path = (
        config.cache_dir / f"corrected_primary_{int(threshold)}_{config.date_tag}.parquet"
    )
    legacy_size_path = (
        config.cache_dir / f"corrected_size_disagreement_{int(threshold)}_{config.date_tag}.parquet"
    )
    legacy_endpoint_path = (
        config.cache_dir / f"corrected_endpoint_symbols_{int(threshold)}_{config.date_tag}.parquet"
    )
    if primary_path.exists() and size_path.exists() and endpoint_path.exists():
        bars = pd.read_parquet(primary_path)
        bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
        bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
        return bars, pd.read_parquet(size_path), pd.read_parquet(endpoint_path)
    if legacy_primary_path.exists() and legacy_size_path.exists() and legacy_endpoint_path.exists():
        bars = pd.read_parquet(legacy_primary_path)
        bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
        bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
        return bars, pd.read_parquet(legacy_size_path), pd.read_parquet(legacy_endpoint_path)

    result = _stream_build_bars(
        sorted_path,
        roots,
        threshold,
        size_thresholds=size_thresholds,
        include_size=True,
        include_endpoint=True,
    )
    complete_index = result.bars.index[result.bars["complete"]]
    bars = result.bars.loc[complete_index].reset_index(drop=True)
    size_disagreement = result.size_disagreement.loc[complete_index].reset_index(drop=True)
    endpoint_symbols = result.endpoint_symbols.loc[complete_index].reset_index(drop=True)
    bars.to_parquet(primary_path, index=False)
    size_disagreement.to_parquet(size_path, index=False)
    endpoint_symbols.to_parquet(endpoint_path, index=False)
    return bars, size_disagreement, endpoint_symbols


def _continuous_price_panel(
    config: MetalsFlowConfig,
    bars: pd.DataFrame,
    roots: tuple[str, ...],
) -> dict[str, pd.DataFrame | pd.Series]:
    if config.continuous_dir is None:
        raise ValueError("continuous_dir is required")
    continuous_by_root = {
        root: pd.read_parquet(config.continuous_dir / f"{root}.parquet") for root in roots
    }
    return align_continuous_marks_to_bars(
        continuous_by_root,
        bars,
        roots,
        max_staleness_seconds=3600.0,
        roll_cooldown_bars=1,
    )


def _trade_size_summary_from_thresholds(
    sorted_path: Path,
    roots: tuple[str, ...],
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    con = duckdb.connect()
    try:
        for row in thresholds.itertuples(index=False):
            root = roots[int(row.root_code)]
            summary = con.execute(
                f"""
                WITH bucketed AS (
                    SELECT
                        CASE
                            WHEN size <= {float(row.q50_size)} THEN 'small'
                            WHEN size <= {float(row.q90_size)} THEN 'medium'
                            WHEN size <= {float(row.q99_size)} THEN 'large'
                            ELSE 'very_large'
                        END AS size_bucket,
                        notional
                    FROM read_parquet('{sorted_path}')
                    WHERE root_code = {int(row.root_code)}
                )
                SELECT
                    '{root}' AS root,
                    size_bucket,
                    count(*) AS trades,
                    sum(notional) AS notional,
                    median(notional) AS median_notional,
                    avg(notional) AS mean_notional
                FROM bucketed
                GROUP BY size_bucket
                ORDER BY size_bucket
                """
            ).fetchdf()
            rows.append(summary)
    finally:
        con.close()
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _write_streaming_artifacts(
    *,
    config: MetalsFlowConfig,
    primary_bars: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    shares: pd.DataFrame,
    features: pd.DataFrame,
    zscores: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_thresholds: pd.DataFrame,
    size_summary: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    endpoint_symbols: pd.DataFrame,
    price_validity: pd.DataFrame,
    continuous_marks: pd.DataFrame,
    fair_value_residuals: pd.DataFrame,
    fair_value_zscores: pd.DataFrame,
    cointegration: pd.DataFrame,
    geometry: Any,
    forward_studies: pd.DataFrame,
    ic: pd.DataFrame,
    classifications: pd.DataFrame,
    event_summary: pd.DataFrame,
    book_availability: pd.DataFrame,
    trade_inventory: pd.DataFrame,
    threshold_summary: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "bars": config.output_dir / "primary_bars.parquet",
        "prices": config.output_dir / "bar_continuous_prices.parquet",
        "returns": config.output_dir / "bar_returns.parquet",
        "shares": config.output_dir / "contribution_shares.parquet",
        "features": config.output_dir / "flow_features.parquet",
        "zscores": config.output_dir / "flow_feature_zscores.parquet",
        "anomalies": config.output_dir / "flow_anomalies.parquet",
        "size_thresholds": config.output_dir / "size_thresholds.csv",
        "trade_size_summary": config.output_dir / "trade_size_summary.csv",
        "trade_size_disagreement": config.output_dir / "trade_size_disagreement.parquet",
        "endpoint_symbols": config.output_dir / "endpoint_symbols.parquet",
        "price_validity": config.output_dir / "price_validity.parquet",
        "continuous_marks": config.output_dir / "continuous_marks.parquet",
        "fair_value_residuals": config.output_dir / "fair_value_residuals.parquet",
        "fair_value_zscores": config.output_dir / "fair_value_zscores.parquet",
        "cointegration": config.output_dir / "cointegration.csv",
        "forward_studies": config.output_dir / "forward_studies.csv",
        "information_coefficients": config.output_dir / "information_coefficients.csv",
        "signal_classification": config.output_dir / "signal_classification.csv",
        "event_study_summary": config.output_dir / "event_study_summary.csv",
        "book_availability": config.output_dir / "mbp1_availability.csv",
        "trade_inventory": config.output_dir / "trade_inventory.csv",
        "threshold_summary": config.output_dir / "threshold_summary.csv",
    }
    primary_bars.to_parquet(paths["bars"], index=False)
    prices.to_parquet(paths["prices"], index=False)
    returns.to_parquet(paths["returns"], index=False)
    shares.to_parquet(paths["shares"], index=False)
    features.to_parquet(paths["features"], index=False)
    zscores.to_parquet(paths["zscores"], index=False)
    anomalies.to_parquet(paths["anomalies"], index=False)
    size_thresholds.to_csv(paths["size_thresholds"], index=False)
    size_summary.to_csv(paths["trade_size_summary"], index=False)
    size_disagreement.to_parquet(paths["trade_size_disagreement"], index=False)
    endpoint_symbols.to_parquet(paths["endpoint_symbols"], index=False)
    price_validity.to_parquet(paths["price_validity"], index=False)
    continuous_marks.to_parquet(paths["continuous_marks"], index=False)
    fair_value_residuals.to_parquet(paths["fair_value_residuals"], index=False)
    fair_value_zscores.to_parquet(paths["fair_value_zscores"], index=False)
    cointegration.to_csv(paths["cointegration"], index=False)
    forward_studies.to_csv(paths["forward_studies"], index=False)
    ic.to_csv(paths["information_coefficients"], index=False)
    classifications.to_csv(paths["signal_classification"], index=False)
    event_summary.to_csv(paths["event_study_summary"], index=False)
    book_availability.to_csv(paths["book_availability"], index=False)
    trade_inventory.to_csv(paths["trade_inventory"], index=False)
    threshold_summary.to_csv(paths["threshold_summary"], index=False)

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


def _summary_payload(
    *,
    config: MetalsFlowConfig,
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
    price_validity: pd.DataFrame,
    trade_inventory: pd.DataFrame,
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
        "trade_inventory": trade_inventory.to_dict(orient="records"),
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
        "price_validity": {
            "valid_price_fraction": float(price_validity["valid_price_mask"].mean()),
            "fresh_all_fraction": float(price_validity["fresh_all"].mean()),
            "roll_invalid_fraction": float(price_validity["roll_invalid"].mean()),
        },
        "mbp1_availability": book_availability.to_dict(orient="records"),
        "geometry_skipped": geometry_skipped,
    }


def json_safe(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


if __name__ == "__main__":
    main()
