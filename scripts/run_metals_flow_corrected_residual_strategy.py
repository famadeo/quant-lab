# ruff: noqa: PLR0911, PLR0915, PLR2004
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantlab.metals_flow.anomaly import anomaly_flags, mahalanobis_distances
from quantlab.metals_flow.config import CONTRACT_MULTIPLIERS, MetalsFlowConfig
from quantlab.metals_flow.data_quality import (
    align_continuous_marks_to_bars,
    bar_log_returns_with_validity,
)
from quantlab.metals_flow.dollar_bars import summarize_bars
from quantlab.metals_flow.fair_value import ewma_relative_value_residuals
from quantlab.metals_flow.features import assemble_flow_features
from quantlab.metals_flow.forward import future_returns, information_coefficients
from quantlab.metals_flow.strategy import (
    backtest_positions,
    benjamini_hochberg,
    calculate_strategy_metrics,
    convergence_state_positions,
    daily_block_bootstrap,
    demean_and_normalize_positions,
    estimate_mbp1_costs,
    monthly_information_coefficients,
)


@dataclass
class StreamBuildResult:
    bars: pd.DataFrame
    size_disagreement: pd.DataFrame | None = None
    endpoint_symbols: pd.DataFrame | None = None


@dataclass(frozen=True)
class CorrectedDataConfig:
    continuous_dir: Path
    output_dir: Path
    size_threshold_calibration_days: int
    max_price_staleness_seconds: float
    roll_cooldown_bars: int
    threshold_window: int
    threshold_min_periods: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected metals flow-filtered residual strategy validation."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Experiment config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    config_path = parse_args().config
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = MetalsFlowConfig.from_mapping(payload, config_path.parent)
    corrected = _corrected_config(payload, config)
    out_dir = corrected.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    lock_path = out_dir / "run.lock"
    lock_fd = _acquire_lock(lock_path)

    try:
        _run(payload, config, corrected, out_dir, plot_dir)
    finally:
        _release_lock(lock_fd, lock_path)


def _corrected_config(payload: dict[str, Any], config: MetalsFlowConfig) -> CorrectedDataConfig:
    data = payload.get("data", {})
    strategy = payload.get("strategy", {})
    outputs = payload.get("outputs", {})
    return CorrectedDataConfig(
        continuous_dir=Path(data["continuous_dir"]),
        output_dir=Path(outputs["directory"]),
        size_threshold_calibration_days=int(strategy.get("size_threshold_calibration_days", 30)),
        max_price_staleness_seconds=float(strategy.get("max_price_staleness_seconds", 3600.0)),
        roll_cooldown_bars=int(strategy.get("roll_cooldown_bars", 1)),
        threshold_window=int(strategy.get("threshold_window", 20_000)),
        threshold_min_periods=int(strategy.get("threshold_min_periods", 2_000)),
    )


def _run(
    payload: dict[str, Any],
    config: MetalsFlowConfig,
    corrected: CorrectedDataConfig,
    out_dir: Path,
    plot_dir: Path,
) -> None:
    roots = tuple(payload["universe"]["roots"])
    core_roots = tuple(payload["universe"]["core_roots"])
    strategy_cfg = payload["strategy"]
    _assert_memory_available(min_gb=2.0, stage="startup")

    print("building symbol-preserving sorted trade cache", flush=True)
    sorted_core_path = _build_sorted_trade_cache_with_symbols(config, core_roots)
    trade_inventory = _trade_inventory_extended(config, roots)
    size_thresholds = _size_thresholds_from_calibration(
        sorted_core_path,
        core_roots,
        start=config.start,
        calibration_days=corrected.size_threshold_calibration_days,
    )

    print("building cross-sectional flow bars", flush=True)
    bars_by_threshold: dict[float, pd.DataFrame] = {}
    for threshold_value in config.thresholds:
        threshold = float(threshold_value)
        cache_path = (
            config.cache_dir
            / f"corrected_bars_core_{int(threshold)}_{config.date_tag}.parquet"
        )
        if cache_path.exists():
            bars = pd.read_parquet(cache_path)
            bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
            bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
        else:
            _assert_memory_available(min_gb=1.0, stage=f"threshold {threshold:,.0f}")
            bars = _stream_build_bars(
                sorted_core_path,
                core_roots,
                threshold,
                size_thresholds=None,
                include_size=False,
                include_endpoint=False,
            ).bars
            bars.to_parquet(cache_path, index=False)
        bars_by_threshold[threshold] = bars
        print(f"  threshold {threshold:,.0f}: {len(bars):,} bars", flush=True)
    threshold_summary = summarize_bars(bars_by_threshold)

    primary = float(config.primary_threshold)
    primary_cache = (
        config.cache_dir / f"corrected_primary_{int(primary)}_{config.date_tag}.parquet"
    )
    size_cache = (
        config.cache_dir
        / f"corrected_size_disagreement_{int(primary)}_{config.date_tag}.parquet"
    )
    endpoint_cache = (
        config.cache_dir / f"corrected_endpoint_symbols_{int(primary)}_{config.date_tag}.parquet"
    )
    if primary_cache.exists() and size_cache.exists() and endpoint_cache.exists():
        bars = pd.read_parquet(primary_cache)
        bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
        bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
        size_disagreement = pd.read_parquet(size_cache)
        endpoint_symbols = pd.read_parquet(endpoint_cache)
    else:
        _assert_memory_available(min_gb=1.0, stage="primary bars")
        primary_result = _stream_build_bars(
            sorted_core_path,
            core_roots,
            primary,
            size_thresholds=size_thresholds,
            include_size=True,
            include_endpoint=True,
        )
        complete_index = primary_result.bars.index[primary_result.bars["complete"]]
        bars = primary_result.bars.loc[complete_index].reset_index(drop=True)
        size_disagreement = primary_result.size_disagreement.loc[complete_index].reset_index(
            drop=True
        )
        endpoint_symbols = primary_result.endpoint_symbols.loc[complete_index].reset_index(
            drop=True
        )
        bars.to_parquet(primary_cache, index=False)
        size_disagreement.to_parquet(size_cache, index=False)
        endpoint_symbols.to_parquet(endpoint_cache, index=False)

    print("aligning continuous prices to dollar bars", flush=True)
    price_panel = _load_continuous_price_panel(
        corrected.continuous_dir,
        bars,
        core_roots,
        max_staleness_seconds=corrected.max_price_staleness_seconds,
        roll_cooldown_bars=corrected.roll_cooldown_bars,
    )
    log_prices = price_panel["log_prices"]
    returns = bar_log_returns_with_validity(log_prices, price_panel["valid_price_mask"])

    print("computing flow features and residuals", flush=True)
    features, shares, _ = assemble_flow_features(
        bars,
        core_roots,
        rolling_window=config.rolling_window,
        min_periods=config.min_periods,
    )
    md_ewma = mahalanobis_distances(
        shares,
        method="ewma",
        min_periods=config.min_periods,
        ewma_halflife=config.ewma_halflife,
    ).rename("md_rolling")
    anomalies = pd.concat(
        [
            md_ewma,
            md_ewma.rename("md_ewma"),
            md_ewma.rename("md_expanding"),
            anomaly_flags(md_ewma),
        ],
        axis=1,
    )
    _, residual_z = ewma_relative_value_residuals(
        log_prices,
        halflife=config.ewma_halflife,
        min_periods=config.fair_value_min_periods,
        zscore_window=config.fair_value_lookback,
    )
    residual_z = residual_z.where(price_panel["valid_price_mask"], axis=0)

    print("running leak-controlled strategy", flush=True)
    md_threshold = _shifted_rolling_quantile(
        anomalies["md_rolling"],
        float(strategy_cfg["md_quantile"]),
        corrected.threshold_window,
        corrected.threshold_min_periods,
    )
    disagreement_threshold = _shifted_rolling_quantile(
        size_disagreement["large_small_l1_distance"],
        float(strategy_cfg["large_small_quantile"]),
        corrected.threshold_window,
        corrected.threshold_min_periods,
    )
    valid_mask = price_panel["valid_price_mask"].fillna(False)
    entry_mask = (
        (anomalies["md_rolling"] >= md_threshold)
        & (size_disagreement["large_small_l1_distance"] >= disagreement_threshold)
        & valid_mask
    )
    root_entry_masks = _root_large_small_masks_walk_forward(
        size_disagreement,
        core_roots,
        quantile=float(strategy_cfg["large_small_quantile"]),
        window=corrected.threshold_window,
        min_periods=corrected.threshold_min_periods,
    ).where(valid_mask, False)
    raw_positions = convergence_state_positions(
        residual_z,
        entry_mask,
        root_entry_masks,
        entry_z=float(strategy_cfg["entry_z"]),
        exit_z=float(strategy_cfg["exit_z"]),
        stop_z=float(strategy_cfg["stop_z"]),
    )
    raw_positions = raw_positions.where(valid_mask, 0.0, axis=0)
    positions = demean_and_normalize_positions(raw_positions)

    cost_estimates = estimate_mbp1_costs(
        config.mbp1_dir,
        core_roots,
        fallback_bps=float(strategy_cfg["cost_fallback_bps"]),
    )
    cost_bps = cost_estimates.set_index("root")["per_side_cost_bps"].reindex(core_roots)
    periods_per_year = _periods_per_year(bars)
    variants, strategy_returns = _run_cost_variants(
        positions,
        returns,
        cost_bps,
        cost_multipliers=tuple(float(x) for x in strategy_cfg["cost_multipliers"]),
        periods_per_year=periods_per_year,
    )
    selected_returns = strategy_returns["cost_1.0"].copy()

    print("running validation diagnostics", flush=True)
    future = future_returns(returns, config.horizons)
    validation_horizon = int(strategy_cfg["validation_horizon"])
    horizon_returns = future.xs(validation_horizon, axis=1, level="horizon")
    feature_panel = _validation_feature_panel(
        residual_z,
        anomalies,
        size_disagreement,
        features,
        core_roots,
    )
    ic = information_coefficients(feature_panel, future)
    ic_fdr = benjamini_hochberg(ic) if not ic.empty else ic
    monthly_ic = monthly_information_coefficients(
        feature_panel,
        horizon_returns,
        bars["end_ts"],
    )
    bootstrap = daily_block_bootstrap(
        selected_returns[["gross_return", "cost_return", "net_return"]],
        bars["end_ts"],
        iterations=int(strategy_cfg["bootstrap_iterations"]),
    )
    split_metrics = purged_split_strategy_metrics(
        selected_returns,
        positions,
        bars["end_ts"],
        train_fraction=float(strategy_cfg["train_fraction"]),
        embargo_bars=int(strategy_cfg["embargo_bars"]),
        periods_per_year=periods_per_year,
    )
    monthly_returns = _monthly_strategy_returns(selected_returns, bars["end_ts"])
    daily_hac = _daily_hac_summary(selected_returns, bars["end_ts"])
    leakage_diagnostics = _threshold_diagnostics(
        anomalies,
        size_disagreement,
        md_threshold,
        disagreement_threshold,
        entry_mask,
        bars["end_ts"],
    )

    print("writing artifacts", flush=True)
    _write_tables(
        out_dir=out_dir,
        threshold_summary=threshold_summary,
        size_thresholds=size_thresholds,
        bars=bars,
        endpoint_symbols=endpoint_symbols,
        continuous_marks=price_panel["continuous_marks"],
        price_validity=price_panel["price_validity"],
        returns=returns,
        shares=shares,
        features=features,
        anomalies=anomalies,
        residual_z=residual_z,
        size_disagreement=size_disagreement,
        positions=positions,
        raw_positions=raw_positions,
        selected_returns=selected_returns,
        variants=variants,
        cost_estimates=cost_estimates,
        ic=ic,
        ic_fdr=ic_fdr,
        monthly_ic=monthly_ic,
        bootstrap=bootstrap,
        split_metrics=split_metrics,
        monthly_returns=monthly_returns,
        daily_hac=daily_hac,
        trade_inventory=trade_inventory,
        leakage_diagnostics=leakage_diagnostics,
    )
    _write_plots(
        plot_dir=plot_dir,
        bars=bars,
        returns=selected_returns,
        positions=positions,
        variants=variants,
        residual_z=residual_z,
        anomalies=anomalies,
        size_disagreement=size_disagreement,
        md_threshold=md_threshold,
        disagreement_threshold=disagreement_threshold,
        monthly_returns=monthly_returns,
        ic_fdr=ic_fdr,
        price_validity=price_panel["price_validity"],
    )

    summary = _summary(
        payload=payload,
        config=config,
        corrected=corrected,
        roots=roots,
        core_roots=core_roots,
        trade_inventory=trade_inventory,
        bars=bars,
        threshold_summary=threshold_summary,
        size_thresholds=size_thresholds,
        price_validity=price_panel["price_validity"],
        features=features,
        anomalies=anomalies,
        size_disagreement=size_disagreement,
        residual_z=residual_z,
        variants=variants,
        selected_returns=selected_returns,
        cost_estimates=cost_estimates,
        ic_fdr=ic_fdr,
        monthly_ic=monthly_ic,
        bootstrap=bootstrap,
        split_metrics=split_metrics,
        monthly_returns=monthly_returns,
        daily_hac=daily_hac,
        leakage_diagnostics=leakage_diagnostics,
        periods_per_year=periods_per_year,
    )
    result_payload = _json_safe(
        {
            "experiment_id": payload["experiment"]["id"],
            "title": payload["experiment"]["title"],
            "completed_at": datetime.now(UTC).isoformat(),
            "summary": summary,
        }
    )
    (out_dir / "results.json").write_text(
        json.dumps(result_payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    _write_report(out_dir, payload, summary)
    print(f"wrote {out_dir / 'results.json'}", flush=True)


def _acquire_lock(lock_path: Path) -> int:
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Refusing to start because {lock_path} exists. "
            "Another run may be active; remove the lock only after confirming no run "
            "process exists."
        ) from exc
    os.write(lock_fd, f"pid={os.getpid()}\nstarted={datetime.now(UTC).isoformat()}\n".encode())
    return lock_fd


def _release_lock(lock_fd: int, lock_path: Path) -> None:
    os.close(lock_fd)
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()


def _build_sorted_trade_cache_with_symbols(
    config: MetalsFlowConfig,
    roots: tuple[str, ...],
) -> Path:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    root_tag = "_".join(roots)
    output = config.cache_dir / f"sorted_trades_with_symbols_{root_tag}_{config.date_tag}.parquet"
    if output.exists():
        return output

    selects = []
    for code, root in enumerate(roots):
        path = config.trade_dir / f"{root}.parquet"
        multiplier = CONTRACT_MULTIPLIERS[root]
        selects.append(
            f"""
            SELECT
                ts_event,
                {code}::UTINYINT AS root_code,
                symbol::VARCHAR AS symbol,
                price::DOUBLE AS price,
                size::DOUBLE AS size,
                (price::DOUBLE * size::DOUBLE * {multiplier}) AS notional,
                CASE
                    WHEN side = 'B' THEN price::DOUBLE * size::DOUBLE * {multiplier}
                    WHEN side = 'A' THEN -price::DOUBLE * size::DOUBLE * {multiplier}
                    ELSE 0.0
                END AS signed_notional
            FROM read_parquet('{path}')
            WHERE ts_event >= TIMESTAMPTZ '{config.start}'
              AND ts_event < TIMESTAMPTZ '{config.end}'
            """
        )
    sql = " UNION ALL ".join(selects)
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='8GB'")
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM ({sql})
            ORDER BY ts_event, root_code, symbol
        )
        TO '{output}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
    )
    con.close()
    return output


def _trade_inventory_extended(config: MetalsFlowConfig, roots: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    con = duckdb.connect()
    for root in roots:
        path = config.trade_dir / f"{root}.parquet"
        multiplier = CONTRACT_MULTIPLIERS[root]
        row = con.execute(
            f"""
            WITH base AS (
                SELECT *
                FROM read_parquet('{path}')
                WHERE ts_event >= TIMESTAMPTZ '{config.start}'
                  AND ts_event < TIMESTAMPTZ '{config.end}'
            ),
            duplicate_groups AS (
                SELECT
                    ts_event,
                    symbol,
                    price,
                    size,
                    side,
                    count(*) AS rows_in_group
                FROM base
                GROUP BY ts_event, symbol, price, size, side
                HAVING count(*) > 1
            )
            SELECT
                '{root}' AS root,
                count(*) AS trades,
                min(ts_event) AS start_ts,
                max(ts_event) AS end_ts,
                count(DISTINCT symbol) AS symbol_count,
                sum(price::DOUBLE * size::DOUBLE * {multiplier}) AS notional,
                sum(CASE
                    WHEN side = 'B' THEN price::DOUBLE * size::DOUBLE * {multiplier}
                    WHEN side = 'A' THEN -price::DOUBLE * size::DOUBLE * {multiplier}
                    ELSE 0.0
                END) / nullif(sum(price::DOUBLE * size::DOUBLE * {multiplier}), 0) AS signed_ratio,
                sum(CASE
                    WHEN side = 'N' THEN price::DOUBLE * size::DOUBLE * {multiplier}
                    ELSE 0.0
                END) / nullif(sum(price::DOUBLE * size::DOUBLE * {multiplier}), 0)
                    AS side_n_notional_share,
                median(size::DOUBLE) AS median_trade_size,
                quantile_cont(size::DOUBLE, 0.99) AS p99_trade_size,
                median(price::DOUBLE * size::DOUBLE * {multiplier}) AS median_trade_notional,
                quantile_cont(price::DOUBLE * size::DOUBLE * {multiplier}, 0.99)
                    AS p99_trade_notional,
                (SELECT coalesce(sum(rows_in_group - 1), 0) FROM duplicate_groups)
                    AS exact_duplicate_extra_rows
            FROM base
            """
        ).fetchdf()
        rows.append(row)
    con.close()
    return pd.concat(rows, ignore_index=True)


def _size_thresholds_from_calibration(
    sorted_path: Path,
    roots: tuple[str, ...],
    *,
    start: str,
    calibration_days: int,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    start_ts = start_ts.tz_convert("UTC") if start_ts.tzinfo else pd.Timestamp(start, tz="UTC")
    end_ts = start_ts + pd.Timedelta(days=calibration_days)
    con = duckdb.connect()
    frame = con.execute(
        f"""
        SELECT
            root_code,
            count(*) AS calibration_trades,
            min(ts_event) AS calibration_start,
            max(ts_event) AS calibration_end,
            quantile_cont(size, 0.50) AS q50_size,
            quantile_cont(size, 0.90) AS q90_size,
            quantile_cont(size, 0.99) AS q99_size
        FROM read_parquet('{sorted_path}')
        WHERE ts_event >= TIMESTAMPTZ '{start_ts.isoformat()}'
          AND ts_event < TIMESTAMPTZ '{end_ts.isoformat()}'
        GROUP BY root_code
        ORDER BY root_code
        """
    ).fetchdf()
    con.close()
    frame["root"] = [roots[int(code)] for code in frame["root_code"]]
    return frame


def _stream_build_bars(
    sorted_path: Path,
    roots: tuple[str, ...],
    threshold: float,
    *,
    size_thresholds: pd.DataFrame | None,
    include_size: bool,
    include_endpoint: bool,
    batch_size: int = 2_000_000,
) -> StreamBuildResult:
    columns = ["ts_event", "root_code", "symbol", "price", "size", "notional", "signed_notional"]
    parquet = pq.ParquetFile(sorted_path)
    n_roots = len(roots)
    rows: list[dict[str, object]] = []
    size_rows: list[dict[str, float]] = []
    endpoint_rows: list[dict[str, object]] = []
    last_symbol = np.full(n_roots, None, dtype=object)
    last_raw_price = np.full(n_roots, np.nan)
    last_ts: pd.Timestamp | None = None

    q50 = np.full(n_roots, np.nan)
    q90 = np.full(n_roots, np.nan)
    q99 = np.full(n_roots, np.nan)
    if size_thresholds is not None:
        for row in size_thresholds.itertuples(index=False):
            q50[int(row.root_code)] = float(row.q50_size)
            q90[int(row.root_code)] = float(row.q90_size)
            q99[int(row.root_code)] = float(row.q99_size)

    state = _empty_bar_state(n_roots)
    state["start_ts"] = None
    bar_id = 0
    rows_seen = 0
    for batch_number, batch in enumerate(
        parquet.iter_batches(batch_size=batch_size, columns=columns),
        start=1,
    ):
        name_to_index = {name: i for i, name in enumerate(columns)}
        arrays = {
            name: batch.column(name_to_index[name]).to_numpy(zero_copy_only=False)
            for name in columns
        }
        arrays["root_code"] = arrays["root_code"].astype(np.int64, copy=False)
        arrays["price"] = arrays["price"].astype(float, copy=False)
        arrays["size"] = arrays["size"].astype(float, copy=False)
        arrays["notional"] = arrays["notional"].astype(float, copy=False)
        arrays["signed_notional"] = arrays["signed_notional"].astype(float, copy=False)
        batch_cumulative = np.cumsum(arrays["notional"])
        prefixes = _batch_prefixes(arrays, n_roots, q50, q90, q99, include_size)
        root_positions = [np.flatnonzero(arrays["root_code"] == code) for code in range(n_roots)]
        pos = 0
        n_rows = len(arrays["notional"])
        rows_seen += n_rows
        last_ts = pd.Timestamp(arrays["ts_event"][-1])
        while pos < n_rows:
            if state["start_ts"] is None:
                state["start_ts"] = pd.Timestamp(arrays["ts_event"][pos])
            remaining = threshold - float(state["bar_notional"])
            prior_batch_notional = batch_cumulative[pos - 1] if pos > 0 else 0.0
            cut_target = prior_batch_notional + remaining
            end = int(np.searchsorted(batch_cumulative, cut_target, side="left"))
            if end >= n_rows:
                _add_prefix_slice_to_state(state, prefixes, pos, n_rows - 1)
                _update_last_trade_from_positions(
                    last_symbol,
                    last_raw_price,
                    arrays,
                    root_positions,
                    pos,
                    n_rows - 1,
                )
                break
            _add_prefix_slice_to_state(state, prefixes, pos, end)
            _update_last_trade_from_positions(
                last_symbol,
                last_raw_price,
                arrays,
                root_positions,
                pos,
                end,
            )
            end_ts = pd.Timestamp(arrays["ts_event"][end])
            rows.append(_bar_record_from_state(bar_id, state, roots, threshold, end_ts, True))
            if include_size:
                size_rows.append(_size_record_from_state(state, roots))
            if include_endpoint:
                endpoint_rows.append(
                    _endpoint_record_from_state(last_symbol, last_raw_price, roots)
                )
            bar_id += 1
            state = _empty_bar_state(n_roots)
            state["start_ts"] = None
            pos = end + 1
        if batch_number == 1 or batch_number % 5 == 0:
            print(
                f"    threshold {threshold:,.0f}: processed {rows_seen:,} trades, "
                f"{len(rows):,} bars, available_mem={_available_memory_gb():.1f} GiB",
                flush=True,
            )

    if state["trade_count"] > 0:
        rows.append(_bar_record_from_state(bar_id, state, roots, threshold, last_ts, False))
        if include_size:
            size_rows.append(_size_record_from_state(state, roots))
        if include_endpoint:
            endpoint_rows.append(_endpoint_record_from_state(last_symbol, last_raw_price, roots))

    return StreamBuildResult(
        bars=pd.DataFrame(rows),
        size_disagreement=pd.DataFrame(size_rows) if include_size else None,
        endpoint_symbols=pd.DataFrame(endpoint_rows) if include_endpoint else None,
    )


def _batch_prefixes(
    arrays: dict[str, np.ndarray],
    n_roots: int,
    q50: np.ndarray,
    q90: np.ndarray,
    q99: np.ndarray,
    include_size: bool,
) -> dict[str, np.ndarray]:
    root_code = arrays["root_code"]
    trade_size = arrays["size"]
    notional = arrays["notional"]
    signed = arrays["signed_notional"]
    n_rows = len(root_code)
    prefixes = {
        "counts": np.zeros((n_roots, n_rows + 1), dtype=np.int64),
        "notional": np.zeros((n_roots, n_rows + 1), dtype=float),
        "signed": np.zeros((n_roots, n_rows + 1), dtype=float),
        "small": np.zeros((n_roots, n_rows + 1), dtype=float),
        "large": np.zeros((n_roots, n_rows + 1), dtype=float),
        "very_large": np.zeros((n_roots, n_rows + 1), dtype=float),
    }
    if include_size:
        small_mask = trade_size <= q50[root_code]
        large_mask = (trade_size > q90[root_code]) & (trade_size <= q99[root_code])
        very_large_mask = trade_size > q99[root_code]
    else:
        small_mask = large_mask = very_large_mask = np.zeros(n_rows, dtype=bool)

    for code in range(n_roots):
        mask = root_code == code
        prefixes["counts"][code, 1:] = np.cumsum(mask, dtype=np.int64)
        prefixes["notional"][code, 1:] = np.cumsum(np.where(mask, notional, 0.0))
        prefixes["signed"][code, 1:] = np.cumsum(np.where(mask, signed, 0.0))
        if include_size:
            prefixes["small"][code, 1:] = np.cumsum(np.where(mask & small_mask, notional, 0.0))
            prefixes["large"][code, 1:] = np.cumsum(np.where(mask & large_mask, notional, 0.0))
            prefixes["very_large"][code, 1:] = np.cumsum(
                np.where(mask & very_large_mask, notional, 0.0)
            )
    return prefixes


def _add_prefix_slice_to_state(
    state: dict[str, Any],
    prefixes: dict[str, np.ndarray],
    start: int,
    end: int,
) -> None:
    left = start
    right = end + 1
    counts = prefixes["counts"][:, right] - prefixes["counts"][:, left]
    notional = prefixes["notional"][:, right] - prefixes["notional"][:, left]
    signed = prefixes["signed"][:, right] - prefixes["signed"][:, left]
    small = prefixes["small"][:, right] - prefixes["small"][:, left]
    large = prefixes["large"][:, right] - prefixes["large"][:, left]
    very_large = prefixes["very_large"][:, right] - prefixes["very_large"][:, left]

    state["counts"] += counts
    state["notional"] += notional
    state["signed"] += signed
    state["small"] += small
    state["large"] += large
    state["very_large"] += very_large
    state["bar_notional"] += float(notional.sum())
    state["trade_count"] += int(counts.sum())


def _update_last_trade_from_positions(
    last_symbol: np.ndarray,
    last_raw_price: np.ndarray,
    arrays: dict[str, np.ndarray],
    root_positions: list[np.ndarray],
    start: int,
    end: int,
) -> None:
    for code, positions in enumerate(root_positions):
        right = np.searchsorted(positions, end, side="right") - 1
        if right >= 0 and positions[right] >= start:
            position = positions[right]
            last_symbol[code] = str(arrays["symbol"][position])
            last_raw_price[code] = float(arrays["price"][position])


def _empty_bar_state(n_roots: int) -> dict[str, Any]:
    return {
        "bar_notional": 0.0,
        "trade_count": 0,
        "notional": np.zeros(n_roots),
        "signed": np.zeros(n_roots),
        "counts": np.zeros(n_roots, dtype=int),
        "small": np.zeros(n_roots),
        "large": np.zeros(n_roots),
        "very_large": np.zeros(n_roots),
    }


def _bar_record_from_state(
    bar_id: int,
    state: dict[str, Any],
    roots: tuple[str, ...],
    threshold: float,
    end_ts: pd.Timestamp,
    complete: bool,
) -> dict[str, object]:
    bar_notional = float(state["bar_notional"])
    counts = state["counts"]
    shares = state["notional"] / bar_notional if bar_notional > 0 else np.zeros(len(roots))
    trade_shares = counts / state["trade_count"] if state["trade_count"] else np.zeros(len(roots))
    dominant_idx = int(np.argmax(shares))
    start_ts = state["start_ts"]
    record: dict[str, object] = {
        "bar_id": bar_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "bar_notional": bar_notional,
        "trades": int(state["trade_count"]),
        "complete": complete,
        "duration_seconds": max((end_ts - start_ts).total_seconds(), 0.0),
        "overshoot_pct": (bar_notional / threshold) - 1.0,
        "threshold": threshold,
        "dominant_root": roots[dominant_idx],
        "dominant_share": float(shares[dominant_idx]),
        "hhi_notional_share": float(np.square(shares).sum()),
        "hhi_trade_share": float(np.square(trade_shares).sum()),
        "complex_signed_notional_ratio": float(state["signed"].sum() / bar_notional)
        if bar_notional > 0
        else np.nan,
    }
    for i, root in enumerate(roots):
        record[f"{root}_notional"] = float(state["notional"][i])
        record[f"{root}_signed_notional"] = float(state["signed"][i])
        record[f"{root}_trades"] = int(counts[i])
    return record


def _endpoint_record_from_state(
    last_symbol: np.ndarray,
    last_raw_price: np.ndarray,
    roots: tuple[str, ...],
) -> dict[str, object]:
    record: dict[str, object] = {}
    for i, root in enumerate(roots):
        record[f"{root}_flow_endpoint_symbol"] = last_symbol[i]
        record[f"{root}_flow_endpoint_raw_price"] = last_raw_price[i]
    return record


def _size_record_from_state(state: dict[str, Any], roots: tuple[str, ...]) -> dict[str, float]:
    small_total = state["small"].sum()
    large_total = state["large"].sum()
    very_large_total = state["very_large"].sum()
    small = state["small"] / small_total if small_total > 0 else np.zeros(len(roots))
    large = state["large"] / large_total if large_total > 0 else np.zeros(len(roots))
    very_large = (
        state["very_large"] / very_large_total if very_large_total > 0 else np.zeros(len(roots))
    )
    diff = large - small
    very_diff = very_large - small
    record = {f"{root}_large_minus_small_share": float(diff[i]) for i, root in enumerate(roots)}
    record["large_small_l1_distance"] = float(np.abs(diff).sum())
    record["very_large_small_l1_distance"] = float(np.abs(very_diff).sum())
    return record


def _load_continuous_price_panel(
    continuous_dir: Path,
    bars: pd.DataFrame,
    roots: tuple[str, ...],
    *,
    max_staleness_seconds: float,
    roll_cooldown_bars: int,
) -> dict[str, pd.DataFrame | pd.Series]:
    continuous_by_root = {
        root: pd.read_parquet(continuous_dir / f"{root}.parquet") for root in roots
    }
    return align_continuous_marks_to_bars(
        continuous_by_root,
        bars,
        roots,
        max_staleness_seconds=max_staleness_seconds,
        roll_cooldown_bars=roll_cooldown_bars,
    )


def _shifted_rolling_quantile(
    series: pd.Series,
    quantile: float,
    window: int,
    min_periods: int,
) -> pd.Series:
    return series.shift(1).rolling(window=window, min_periods=min_periods).quantile(quantile)


def _root_large_small_masks_walk_forward(
    disagreement: pd.DataFrame,
    roots: tuple[str, ...],
    *,
    quantile: float,
    window: int,
    min_periods: int,
) -> pd.DataFrame:
    masks = pd.DataFrame(False, index=disagreement.index, columns=roots)
    for root in roots:
        column = f"{root}_large_minus_small_share"
        threshold = _shifted_rolling_quantile(disagreement[column], quantile, window, min_periods)
        masks[root] = disagreement[column] >= threshold
    return masks


def _run_cost_variants(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: pd.Series,
    *,
    cost_multipliers: tuple[float, ...],
    periods_per_year: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    frames = {}
    for multiplier in cost_multipliers:
        label = f"cost_{multiplier:.1f}"
        frame, metrics = backtest_positions(
            positions,
            returns,
            cost_bps * multiplier,
            periods_per_year=periods_per_year,
        )
        frame["variant"] = label
        frames[label] = frame
        rows.append({"variant": label, "cost_multiplier": multiplier, **metrics.to_dict()})
    return pd.DataFrame(rows), frames


def _validation_feature_panel(
    residual_z: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    features: pd.DataFrame,
    roots: tuple[str, ...],
) -> pd.DataFrame:
    parts = [
        residual_z.add_prefix("rv_z_"),
        anomalies[["md_rolling", "md_expanding", "md_ewma"]],
        size_disagreement,
        features[["entropy_normalized", "hhi", "distance_from_equal_weight"]],
    ]
    parts.extend(
        features[[f"{root}_signed_notional_share", f"{root}_notional_share"]] for root in roots
    )
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def _monthly_strategy_returns(returns: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    frame = returns.loc[:, ["gross_return", "cost_return", "net_return", "turnover"]].copy()
    frame["month"] = pd.to_datetime(timestamps, utc=True).dt.to_period("M").astype(str)
    return (
        frame.groupby("month", as_index=False)
        .agg(
            gross_return=("gross_return", "sum"),
            cost_return=("cost_return", "sum"),
            net_return=("net_return", "sum"),
            turnover=("turnover", "sum"),
            active_bars=("net_return", lambda values: int((values != 0.0).sum())),
        )
        .sort_values("month")
    )


def purged_split_strategy_metrics(
    returns: pd.DataFrame,
    positions: pd.DataFrame,
    timestamps: pd.Series,
    *,
    train_fraction: float,
    embargo_bars: int,
    periods_per_year: float,
) -> pd.DataFrame:
    ordered = pd.to_datetime(timestamps, utc=True).reset_index(drop=True)
    split_index = int(len(ordered) * train_fraction)
    split_index = min(max(split_index, 1), len(ordered) - 1)
    test_start = min(split_index + embargo_bars, len(returns))
    applied_positions = positions.shift(1).fillna(0.0)
    while test_start < len(returns):
        current_gross = positions.iloc[test_start].abs().sum()
        applied_gross = applied_positions.iloc[test_start].abs().sum()
        if current_gross == 0.0 and applied_gross == 0.0:
            break
        test_start += 1

    rows = []
    for label, frame in (
        ("train", returns.iloc[:split_index]),
        ("test_purged", returns.iloc[test_start:]),
    ):
        metrics = calculate_strategy_metrics(frame, periods_per_year)
        rows.append(
            {
                "split": label,
                "start_ts": ordered.iloc[0] if label == "train" else ordered.iloc[test_start],
                "end_ts": ordered.iloc[split_index - 1] if label == "train" else ordered.iloc[-1],
                "split_index": split_index,
                "test_start_index": test_start,
                **metrics.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def _daily_hac_summary(returns: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    days = pd.to_datetime(timestamps, utc=True).dt.date
    daily = returns.assign(day=days).groupby("day", as_index=False)["net_return"].sum()
    values = daily["net_return"].to_numpy(dtype=float)
    rows = [
        {
            "frequency": "daily",
            "lags": lags,
            "observations": len(values),
            "mean": float(np.mean(values)) if len(values) else np.nan,
            "tstat": _newey_west_tstat(values, lags),
            "net_return": float(np.sum(values)) if len(values) else 0.0,
        }
        for lags in (0, 5, 20, 60, 120)
    ]
    return pd.DataFrame(rows)


def _newey_west_tstat(values: np.ndarray, lags: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return np.nan
    centered = values - values.mean()
    gamma0 = float(centered @ centered / n)
    long_run = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        cov = float(centered[lag:] @ centered[:-lag] / n)
        weight = 1.0 - lag / (lags + 1.0)
        long_run += 2.0 * weight * cov
    if long_run <= 0.0:
        return np.nan
    return float(values.mean() / math.sqrt(long_run / n))


def _threshold_diagnostics(
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    md_threshold: pd.Series,
    disagreement_threshold: pd.Series,
    entry_mask: pd.Series,
    timestamps: pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(timestamps, utc=True).dt.to_period("M").astype(str),
            "md": anomalies["md_rolling"],
            "md_threshold": md_threshold,
            "large_small_l1": size_disagreement["large_small_l1_distance"],
            "large_small_threshold": disagreement_threshold,
            "entry": entry_mask,
        }
    )
    return (
        frame.groupby("month", as_index=False)
        .agg(
            bars=("entry", "size"),
            md_threshold_median=("md_threshold", "median"),
            large_small_threshold_median=("large_small_threshold", "median"),
            md_pass_rate=(
                "md",
                lambda values: float(
                    (values >= frame.loc[values.index, "md_threshold"]).mean()
                ),
            ),
            large_small_pass_rate=(
                "large_small_l1",
                lambda values: float(
                    (values >= frame.loc[values.index, "large_small_threshold"]).mean()
                ),
            ),
            entry_rate=("entry", "mean"),
            entries=("entry", "sum"),
        )
        .sort_values("month")
    )


def _write_tables(**kwargs: Any) -> None:
    out_dir = kwargs["out_dir"]
    for name in [
        "bars",
        "endpoint_symbols",
        "continuous_marks",
        "price_validity",
        "returns",
        "shares",
        "features",
        "anomalies",
        "residual_z",
        "size_disagreement",
        "positions",
        "raw_positions",
        "selected_returns",
    ]:
        kwargs[name].to_parquet(out_dir / f"{name}.parquet", index=False)

    for name in [
        "threshold_summary",
        "size_thresholds",
        "variants",
        "cost_estimates",
        "ic",
        "ic_fdr",
        "monthly_ic",
        "bootstrap",
        "split_metrics",
        "monthly_returns",
        "daily_hac",
        "trade_inventory",
        "leakage_diagnostics",
    ]:
        kwargs[name].to_csv(out_dir / f"{name}.csv", index=False)


def _write_plots(
    *,
    plot_dir: Path,
    bars: pd.DataFrame,
    returns: pd.DataFrame,
    positions: pd.DataFrame,
    variants: pd.DataFrame,
    residual_z: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    md_threshold: pd.Series,
    disagreement_threshold: pd.Series,
    monthly_returns: pd.DataFrame,
    ic_fdr: pd.DataFrame,
    price_validity: pd.DataFrame,
) -> None:
    timestamps = pd.to_datetime(bars["end_ts"], utc=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(timestamps, returns["gross_return"].cumsum() * 100, label="gross")
    axes[0].plot(timestamps, returns["net_return"].cumsum() * 100, label="net")
    axes[0].set_title("Corrected flow-filtered residual reversion cumulative return")
    axes[0].set_ylabel("Cumulative log return (%)")
    axes[0].legend()
    equity = returns["net_return"].cumsum()
    drawdown = equity - equity.cummax()
    axes[1].fill_between(timestamps, drawdown * 100, 0, color="tab:red", alpha=0.35)
    axes[1].set_title("Net drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "strategy_equity_drawdown.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    variants.set_index("variant")[["gross_return", "cost_return", "net_return"]].plot.bar(ax=axis)
    axis.set_title("Cost sensitivity")
    axis.set_ylabel("Log return")
    fig.tight_layout()
    fig.savefig(plot_dir / "cost_sensitivity.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(residual_z.columns),
        1,
        figsize=(12, 1.8 * len(residual_z.columns)),
        sharex=True,
    )
    for axis, root in zip(np.ravel(axes), residual_z.columns, strict=True):
        axis.plot(timestamps, residual_z[root], linewidth=0.7)
        axis.axhline(2.0, color="red", linestyle="--", linewidth=0.7)
        axis.axhline(-2.0, color="red", linestyle="--", linewidth=0.7)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_ylabel(root)
    axes[0].set_title("Residual z-scores from continuous price marks")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_zscores.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(timestamps, anomalies["md_rolling"], linewidth=0.7, label="md")
    axes[0].plot(timestamps, md_threshold, linewidth=0.7, label="walk-forward threshold")
    axes[0].set_title("Rolling Mahalanobis distance with shifted threshold")
    axes[0].legend()
    axes[1].plot(
        timestamps,
        size_disagreement["large_small_l1_distance"],
        linewidth=0.7,
        label="large-small L1",
    )
    axes[1].plot(timestamps, disagreement_threshold, linewidth=0.7, label="shifted threshold")
    axes[1].set_title("Large-minus-small flow disagreement with shifted threshold")
    axes[1].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "flow_filters.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(timestamps, price_validity["valid_price_mask"].rolling(1000).mean())
    axes[0].set_title("Rolling valid continuous-price mark fraction")
    axes[1].plot(timestamps, price_validity["roll_any"].astype(float).rolling(1000).sum())
    axes[1].set_title("Rolling active-contract switch count")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "price_validity.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(timestamps, positions.abs().sum(axis=1))
    axis.set_title("Gross exposure after validity masks and convergence-state exits")
    axis.set_ylabel("Gross exposure")
    fig.tight_layout()
    fig.savefig(plot_dir / "gross_exposure.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 4))
    monthly_returns.plot.bar(x="month", y="net_return", ax=axis, legend=False)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Monthly net returns")
    axis.set_ylabel("Log return")
    fig.tight_layout()
    fig.savefig(plot_dir / "monthly_returns.png", dpi=150)
    plt.close(fig)

    if not ic_fdr.empty:
        top_ic = (
            ic_fdr.assign(abs_ic=ic_fdr["spearman_ic"].abs())
            .sort_values("abs_ic", ascending=False)
            .head(20)
        )
        fig, axis = plt.subplots(figsize=(10, 6))
        labels = top_ic["feature"] + "->" + top_ic["root"] + " h" + top_ic["horizon"].astype(str)
        axis.barh(labels[::-1], top_ic["spearman_ic"].iloc[::-1])
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title("Top information coefficients after corrected price marks")
        fig.tight_layout()
        fig.savefig(plot_dir / "top_ic.png", dpi=150)
        plt.close(fig)


def _summary(
    *,
    payload: dict[str, Any],
    config: MetalsFlowConfig,
    corrected: CorrectedDataConfig,
    roots: tuple[str, ...],
    core_roots: tuple[str, ...],
    trade_inventory: pd.DataFrame,
    bars: pd.DataFrame,
    threshold_summary: pd.DataFrame,
    size_thresholds: pd.DataFrame,
    price_validity: pd.DataFrame,
    features: pd.DataFrame,
    anomalies: pd.DataFrame,
    size_disagreement: pd.DataFrame,
    residual_z: pd.DataFrame,
    variants: pd.DataFrame,
    selected_returns: pd.DataFrame,
    cost_estimates: pd.DataFrame,
    ic_fdr: pd.DataFrame,
    monthly_ic: pd.DataFrame,
    bootstrap: pd.DataFrame,
    split_metrics: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    daily_hac: pd.DataFrame,
    leakage_diagnostics: pd.DataFrame,
    periods_per_year: float,
) -> dict[str, Any]:
    selected_metrics = calculate_strategy_metrics(selected_returns, periods_per_year).to_dict()
    top_ic = (
        ic_fdr.assign(abs_ic=ic_fdr["spearman_ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .head(20)
        .to_dict(orient="records")
        if not ic_fdr.empty
        else []
    )
    fdr_significant = int((ic_fdr["bh_qvalue"] <= 0.05).sum()) if not ic_fdr.empty else 0
    month_net = monthly_returns.set_index("month")["net_return"]
    post_2025_10 = float(month_net.loc[month_net.index >= "2025-10"].sum())
    total_net = float(month_net.sum())

    return {
        "status": "corrected_validation",
        "important_note": (
            "Flow features are built from all outright trades. Traded returns and residuals "
            "use 1-minute roll-adjusted continuous marks sampled at dollar-bar ends."
        ),
        "roots": list(roots),
        "core_roots": list(core_roots),
        "start": config.start,
        "end": config.end,
        "continuous_dir": str(corrected.continuous_dir),
        "trade_inventory": trade_inventory.to_dict(orient="records"),
        "primary_complete_bars": len(bars),
        "periods_per_year": periods_per_year,
        "threshold_summary": threshold_summary.to_dict(orient="records"),
        "size_thresholds": size_thresholds.to_dict(orient="records"),
        "size_threshold_calibration_days": corrected.size_threshold_calibration_days,
        "valid_price_fraction": float(price_validity["valid_price_mask"].mean()),
        "roll_invalid_fraction": float(price_validity["roll_invalid"].mean()),
        "fresh_all_fraction": float(price_validity["fresh_all"].mean()),
        "mean_core_contribution": {
            root: float(bars[f"{root}_notional"].sum() / bars["bar_notional"].sum())
            for root in core_roots
        },
        "filter_pass_rate": float((selected_returns["active"]).mean()),
        "selected_strategy_metrics": selected_metrics,
        "cost_variants": variants.replace([np.inf, -np.inf], np.nan).to_dict(orient="records"),
        "cost_estimates": cost_estimates.to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "split_metrics": split_metrics.to_dict(orient="records"),
        "daily_hac": daily_hac.to_dict(orient="records"),
        "monthly_net_positive_fraction": (
            float((month_net > 0).mean()) if len(month_net) else np.nan
        ),
        "post_2025_10_net": post_2025_10,
        "post_2025_10_share_of_total_net": post_2025_10 / total_net if total_net else np.nan,
        "top_information_coefficients": top_ic,
        "fdr_significant_tests": fdr_significant,
        "monthly_ic_summary": monthly_ic.groupby("root")["spearman_ic"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_dict(orient="records")
        if not monthly_ic.empty
        else [],
        "residual_z_abs_mean": residual_z.abs().mean().to_dict(),
        "feature_correlations_with_md": features[
            [
                "entropy_normalized",
                "hhi",
                "distance_from_equal_weight",
                "abs_complex_signed_notional_ratio",
            ]
        ]
        .corrwith(anomalies["md_rolling"])
        .to_dict(),
        "leakage_diagnostics_tail": leakage_diagnostics.tail(12).to_dict(orient="records"),
        "config": payload,
    }


def _write_report(out_dir: Path, payload: dict[str, Any], summary: dict[str, Any]) -> None:
    metrics = summary["selected_strategy_metrics"]
    split = pd.DataFrame(summary["split_metrics"])
    daily_hac = pd.DataFrame(summary["daily_hac"])
    text = f"""---
title: "{payload["experiment"]["id"]} Corrected Validation"
format: html
---

## Status

Status: `corrected_validation`.

This run corrects the HYP-0014 data construction issue by using all-outright
trades only for flow features and 1-minute roll-adjusted continuous prices for
returns and residuals.

## Headline

- Net log return: `{metrics["net_return"]:.6f}`
- Gross log return: `{metrics["gross_return"]:.6f}`
- Cost log return: `{metrics["cost_return"]:.6f}`
- Gross/cost: `{metrics["gross_to_cost"]:.3f}`
- Bar t-stat: `{metrics["tstat"]:.3f}`
- Annualized Sharpe: `{metrics["annualized_sharpe"]:.3f}`
- Max drawdown: `{metrics["max_drawdown"]:.6f}`
- Active bars: `{metrics["active_bars"]:,}`

## Data Validity

- Valid continuous-price mark fraction: `{summary["valid_price_fraction"]:.3%}`
- Roll-invalid fraction: `{summary["roll_invalid_fraction"]:.3%}`
- Fresh-all fraction: `{summary["fresh_all_fraction"]:.3%}`
- Size thresholds are calibrated only from the first
  `{summary["size_threshold_calibration_days"]}` days.
- Flow thresholds use shifted rolling quantiles, not full-sample quantiles.

## Split Metrics

```text
{split.to_string(index=False) if not split.empty else "No split metrics."}
```

## Daily HAC

```text
{daily_hac.to_string(index=False) if not daily_hac.empty else "No HAC metrics."}
```

## Plots

- `plots/strategy_equity_drawdown.png`
- `plots/monthly_returns.png`
- `plots/cost_sensitivity.png`
- `plots/flow_filters.png`
- `plots/price_validity.png`
- `plots/gross_exposure.png`
- `plots/residual_zscores.png`
- `plots/top_ic.png`
"""
    (out_dir / "report.qmd").write_text(text, encoding="utf-8")


def _json_safe(value: Any) -> Any:  # noqa: PLR0912
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bool | str):
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.Series):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        return None
    return value


def _periods_per_year(bars: pd.DataFrame) -> float:
    elapsed_years = (
        pd.to_datetime(bars["end_ts"], utc=True).max()
        - pd.to_datetime(bars["start_ts"], utc=True).min()
    ).total_seconds() / (365.25 * 86_400.0)
    return len(bars) / elapsed_years if elapsed_years > 0 else float(len(bars))


def _available_memory_gb() -> float:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return float("inf")
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return float("inf")


def _assert_memory_available(*, min_gb: float, stage: str) -> None:
    available = _available_memory_gb()
    if available < min_gb:
        raise MemoryError(
            f"Only {available:.2f} GiB available before {stage}; "
            f"need at least {min_gb:.2f} GiB."
        )


if __name__ == "__main__":
    main()
