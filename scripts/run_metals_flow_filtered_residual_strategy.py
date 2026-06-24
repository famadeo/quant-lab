# ruff: noqa: PLR0915, PLR2004
from __future__ import annotations

import argparse
import contextlib
import json
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
from quantlab.metals_flow.dollar_bars import bar_log_returns, summarize_bars
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
    split_strategy_metrics,
)


@dataclass
class StreamBuildResult:
    bars: pd.DataFrame
    prices: pd.DataFrame | None = None
    size_disagreement: pd.DataFrame | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run flow-filtered metals residual mean-reversion strategy."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("experiments/HYP-0013-metals-flow-filtered-residual-reversion/config.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    config_path = parse_args().config
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = MetalsFlowConfig.from_mapping(payload, config_path.parent)
    out_dir = Path(payload["outputs"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    lock_path = out_dir / "run.lock"
    lock_fd = _acquire_lock(lock_path)

    try:
        _run(payload, config, out_dir, plot_dir)
    finally:
        _release_lock(lock_fd, lock_path)


def _run(
    payload: dict[str, Any],
    config: MetalsFlowConfig,
    out_dir: Path,
    plot_dir: Path,
) -> None:
    roots = tuple(payload["universe"]["roots"])
    core_roots = tuple(payload["universe"]["core_roots"])
    isolated_roots = tuple(payload["universe"].get("isolated_roots", []))
    strategy_cfg = payload["strategy"]
    _assert_memory_available(min_gb=2.0, stage="startup")
    print("building sorted core trade cache", flush=True)
    sorted_core_path = _build_sorted_trade_cache(config, core_roots)
    trade_inventory = _trade_inventory(config, roots)
    ali_diagnostics = trade_inventory[trade_inventory["root"].isin(isolated_roots)].copy()
    size_thresholds = _size_thresholds(sorted_core_path, core_roots)

    print("building threshold bars with streaming builder", flush=True)
    bars_by_threshold: dict[float, pd.DataFrame] = {}
    for threshold_value in config.thresholds:
        threshold = float(threshold_value)
        cache_path = (
            config.cache_dir / f"stream_bars_core_{int(threshold)}_{config.date_tag}.parquet"
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
                include_prices=False,
                include_size=False,
            ).bars
            bars.to_parquet(cache_path, index=False)
        bars_by_threshold[threshold] = bars
        print(f"  threshold {threshold:,.0f}: {len(bars):,} bars", flush=True)
    threshold_summary = summarize_bars(bars_by_threshold)

    print("building primary bars, prices, and size disagreement", flush=True)
    primary = float(config.primary_threshold)
    primary_cache = config.cache_dir / f"stream_primary_{int(primary)}_{config.date_tag}.parquet"
    price_cache = config.cache_dir / f"stream_prices_{int(primary)}_{config.date_tag}.parquet"
    size_cache = (
        config.cache_dir / f"stream_size_disagreement_{int(primary)}_{config.date_tag}.parquet"
    )
    if primary_cache.exists() and price_cache.exists() and size_cache.exists():
        bars = pd.read_parquet(primary_cache)
        bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
        bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
        prices = pd.read_parquet(price_cache)
        size_disagreement = pd.read_parquet(size_cache)
    else:
        _assert_memory_available(min_gb=1.0, stage="primary bars/prices/size disagreement")
        primary_result = _stream_build_bars(
            sorted_core_path,
            core_roots,
            primary,
            size_thresholds=size_thresholds,
            include_prices=True,
            include_size=True,
        )
        bars = primary_result.bars.loc[lambda frame: frame["complete"]].reset_index(drop=True)
        prices = primary_result.prices.loc[bars.index].reset_index(drop=True)
        size_disagreement = primary_result.size_disagreement.loc[bars.index].reset_index(drop=True)
        bars.to_parquet(primary_cache, index=False)
        prices.to_parquet(price_cache, index=False)
        size_disagreement.to_parquet(size_cache, index=False)

    print("computing features and residuals", flush=True)
    returns = bar_log_returns(prices).reset_index(drop=True)
    log_prices = np.log(prices.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).ffill()
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
    )
    md_ewma = md_ewma.rename("md_rolling")
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

    print("running strategy and validation", flush=True)
    md_threshold = float(anomalies["md_rolling"].dropna().quantile(strategy_cfg["md_quantile"]))
    disagreement_threshold = float(
        size_disagreement["large_small_l1_distance"]
        .dropna()
        .quantile(strategy_cfg["large_small_quantile"])
    )
    entry_mask = (anomalies["md_rolling"] >= md_threshold) & (
        size_disagreement["large_small_l1_distance"] >= disagreement_threshold
    )
    root_entry_masks = _root_large_small_masks(
        size_disagreement,
        core_roots,
        quantile=float(strategy_cfg["large_small_quantile"]),
    )
    raw_positions = convergence_state_positions(
        residual_z,
        entry_mask,
        root_entry_masks,
        entry_z=float(strategy_cfg["entry_z"]),
        exit_z=float(strategy_cfg["exit_z"]),
        stop_z=float(strategy_cfg["stop_z"]),
    )
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
    ic_fdr = benjamini_hochberg(ic)
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
    split_metrics = split_strategy_metrics(
        selected_returns,
        bars["end_ts"],
        train_fraction=float(strategy_cfg["train_fraction"]),
        embargo_bars=int(strategy_cfg["embargo_bars"]),
        periods_per_year=periods_per_year,
    )
    monthly_returns = _monthly_strategy_returns(selected_returns, bars["end_ts"])

    print("writing artifacts", flush=True)
    _write_tables(
        out_dir=out_dir,
        threshold_summary=threshold_summary,
        bars=bars,
        prices=prices,
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
        trade_inventory=trade_inventory,
        ali_diagnostics=ali_diagnostics,
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
        monthly_returns=monthly_returns,
        monthly_ic=monthly_ic,
        ic_fdr=ic_fdr,
    )

    summary = _summary(
        config=config,
        roots=roots,
        core_roots=core_roots,
        isolated_roots=isolated_roots,
        trade_inventory=trade_inventory,
        bars=bars,
        threshold_summary=threshold_summary,
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
        ali_diagnostics=ali_diagnostics,
        periods_per_year=periods_per_year,
        md_threshold=md_threshold,
        disagreement_threshold=disagreement_threshold,
    )
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "experiment_id": payload["experiment"]["id"],
                "title": payload["experiment"]["title"],
                "completed_at": datetime.now(UTC).isoformat(),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'results.json'}", flush=True)


def _acquire_lock(lock_path: Path) -> int:
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Refusing to start because {lock_path} exists. "
            "Another run may be active; remove the lock only after confirming "
            "no run process exists."
        ) from exc
    os.write(lock_fd, f"pid={os.getpid()}\nstarted={datetime.now(UTC).isoformat()}\n".encode())
    return lock_fd


def _release_lock(lock_fd: int, lock_path: Path) -> None:
    os.close(lock_fd)
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()


def _build_sorted_trade_cache(config: MetalsFlowConfig, roots: tuple[str, ...]) -> Path:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    root_tag = "_".join(roots)
    output = config.cache_dir / f"sorted_trades_{root_tag}_{config.date_tag}.parquet"
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
                price::DOUBLE AS price,
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
            ORDER BY ts_event, root_code
        )
        TO '{output}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000)
        """
    )
    con.close()
    return output


def _trade_inventory(config: MetalsFlowConfig, roots: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    con = duckdb.connect()
    for root in roots:
        path = config.trade_dir / f"{root}.parquet"
        multiplier = CONTRACT_MULTIPLIERS[root]
        row = con.execute(
            f"""
            SELECT
                '{root}' AS root,
                count(*) AS trades,
                min(ts_event) AS start_ts,
                max(ts_event) AS end_ts,
                sum(price::DOUBLE * size::DOUBLE * {multiplier}) AS notional,
                sum(CASE
                    WHEN side = 'B' THEN price::DOUBLE * size::DOUBLE * {multiplier}
                    WHEN side = 'A' THEN -price::DOUBLE * size::DOUBLE * {multiplier}
                    ELSE 0.0
                END) / nullif(sum(price::DOUBLE * size::DOUBLE * {multiplier}), 0) AS signed_ratio,
                median(price::DOUBLE * size::DOUBLE * {multiplier}) AS median_trade_notional,
                quantile_cont(
                    price::DOUBLE * size::DOUBLE * {multiplier}, 0.99
                ) AS p99_trade_notional
            FROM read_parquet('{path}')
            WHERE ts_event >= TIMESTAMPTZ '{config.start}'
              AND ts_event < TIMESTAMPTZ '{config.end}'
            """
        ).fetchdf()
        rows.append(row)
    con.close()
    return pd.concat(rows, ignore_index=True)


def _size_thresholds(sorted_path: Path, roots: tuple[str, ...]) -> pd.DataFrame:
    con = duckdb.connect()
    frame = con.execute(
        f"""
        SELECT
            root_code,
            quantile_cont(notional, 0.50) AS q50,
            quantile_cont(notional, 0.90) AS q90,
            quantile_cont(notional, 0.99) AS q99
        FROM read_parquet('{sorted_path}')
        GROUP BY root_code
        ORDER BY root_code
        """
    ).fetchdf()
    con.close()
    frame["root"] = [roots[int(code)] for code in frame["root_code"]]
    return frame


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
            f"need at least {min_gb:.2f} GiB. Refusing to start a memory-risky step."
        )


def _stream_build_bars(
    sorted_path: Path,
    roots: tuple[str, ...],
    threshold: float,
    *,
    size_thresholds: pd.DataFrame | None,
    include_prices: bool,
    include_size: bool,
    batch_size: int = 2_000_000,
) -> StreamBuildResult:
    columns = ["ts_event", "root_code", "price", "notional", "signed_notional"]
    parquet = pq.ParquetFile(sorted_path)
    n_roots = len(roots)
    rows: list[dict[str, object]] = []
    price_rows: list[dict[str, float]] = []
    size_rows: list[dict[str, float]] = []
    last_price = np.full(n_roots, np.nan)
    last_ts: pd.Timestamp | None = None

    q50 = np.full(n_roots, np.nan)
    q90 = np.full(n_roots, np.nan)
    q99 = np.full(n_roots, np.nan)
    if size_thresholds is not None:
        for row in size_thresholds.itertuples(index=False):
            q50[int(row.root_code)] = float(row.q50)
            q90[int(row.root_code)] = float(row.q90)
            q99[int(row.root_code)] = float(row.q99)

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
                _update_last_price_from_positions(
                    last_price, arrays, root_positions, pos, n_rows - 1
                )
                break
            _add_prefix_slice_to_state(state, prefixes, pos, end)
            _update_last_price_from_positions(last_price, arrays, root_positions, pos, end)
            end_ts = pd.Timestamp(arrays["ts_event"][end])
            rows.append(_bar_record_from_state(bar_id, state, roots, threshold, end_ts, True))
            if include_prices:
                price_rows.append(dict(zip(roots, last_price, strict=True)))
            if include_size:
                size_rows.append(_size_record_from_state(state, roots))
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
        rows.append(
            _bar_record_from_state(
                bar_id,
                state,
                roots,
                threshold,
                last_ts,
                False,
            )
        )
        if include_prices:
            price_rows.append(dict(zip(roots, last_price, strict=True)))
        if include_size:
            size_rows.append(_size_record_from_state(state, roots))

    bars = pd.DataFrame(rows)
    prices = pd.DataFrame(price_rows) if include_prices else None
    size_disagreement = pd.DataFrame(size_rows) if include_size else None
    return StreamBuildResult(bars=bars, prices=prices, size_disagreement=size_disagreement)


def _batch_prefixes(
    arrays: dict[str, np.ndarray],
    n_roots: int,
    q50: np.ndarray,
    q90: np.ndarray,
    q99: np.ndarray,
    include_size: bool,
) -> dict[str, np.ndarray]:
    root_code = arrays["root_code"]
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
        small_mask = notional <= q50[root_code]
        large_mask = (notional > q90[root_code]) & (notional <= q99[root_code])
        very_large_mask = notional > q99[root_code]
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


def _update_last_price_from_positions(
    last_price: np.ndarray,
    arrays: dict[str, np.ndarray],
    root_positions: list[np.ndarray],
    start: int,
    end: int,
) -> None:
    for code, positions in enumerate(root_positions):
        right = np.searchsorted(positions, end, side="right") - 1
        if right >= 0 and positions[right] >= start:
            last_price[code] = float(arrays["price"][positions[right]])


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
    for i, root in enumerate(roots):
        record[f"{root}_signed_notional"] = float(state["signed"][i])
    for i, root in enumerate(roots):
        record[f"{root}_trades"] = int(counts[i])
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


def _root_large_small_masks(
    disagreement: pd.DataFrame,
    roots: tuple[str, ...],
    *,
    quantile: float,
) -> pd.DataFrame:
    masks = pd.DataFrame(False, index=disagreement.index, columns=roots)
    for root in roots:
        column = f"{root}_large_minus_small_share"
        threshold = disagreement[column].dropna().quantile(quantile)
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


def _write_tables(**kwargs: Any) -> None:
    out_dir = kwargs["out_dir"]
    for name in [
        "bars",
        "prices",
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
        "variants",
        "cost_estimates",
        "ic",
        "ic_fdr",
        "monthly_ic",
        "bootstrap",
        "split_metrics",
        "monthly_returns",
        "trade_inventory",
        "ali_diagnostics",
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
    monthly_returns: pd.DataFrame,
    monthly_ic: pd.DataFrame,
    ic_fdr: pd.DataFrame,
) -> None:
    timestamps = pd.to_datetime(bars["end_ts"], utc=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(timestamps, returns["gross_return"].cumsum() * 100, label="gross")
    axes[0].plot(timestamps, returns["net_return"].cumsum() * 100, label="net")
    axes[0].set_title("Flow-filtered residual reversion cumulative return")
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
    axes[0].set_title("Residual z-scores")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "residual_zscores.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(timestamps, anomalies["md_rolling"], linewidth=0.7)
    axes[0].set_title("Rolling Mahalanobis distance")
    axes[1].plot(timestamps, size_disagreement["large_small_l1_distance"], linewidth=0.7)
    axes[1].set_title("Large-minus-small flow disagreement")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_dir / "flow_filters.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 4))
    axis.plot(timestamps, positions.abs().sum(axis=1))
    axis.set_title("Gross exposure after convergence-state exits")
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

    top_ic = (
        ic_fdr.assign(abs_ic=ic_fdr["spearman_ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .head(20)
    )
    fig, axis = plt.subplots(figsize=(10, 6))
    labels = top_ic["feature"] + "->" + top_ic["root"] + " h" + top_ic["horizon"].astype(str)
    axis.barh(labels[::-1], top_ic["spearman_ic"].iloc[::-1])
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_title("Top information coefficients after FDR scan")
    fig.tight_layout()
    fig.savefig(plot_dir / "top_ic.png", dpi=150)
    plt.close(fig)

    if not monthly_ic.empty:
        pivot = monthly_ic.pivot_table(
            index="month",
            columns="root",
            values="spearman_ic",
            aggfunc="mean",
        )
        fig, axis = plt.subplots(figsize=(11, 5))
        image = axis.imshow(pivot.T, aspect="auto", cmap="coolwarm", vmin=-0.15, vmax=0.15)
        axis.set_xticks(range(len(pivot.index)), labels=pivot.index, rotation=90)
        axis.set_yticks(range(len(pivot.columns)), labels=pivot.columns)
        axis.set_title("Monthly mean IC by root")
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        fig.savefig(plot_dir / "monthly_ic_heatmap.png", dpi=150)
        plt.close(fig)


def _summary(
    *,
    config: MetalsFlowConfig,
    roots: tuple[str, ...],
    core_roots: tuple[str, ...],
    isolated_roots: tuple[str, ...],
    trade_inventory: pd.DataFrame,
    bars: pd.DataFrame,
    threshold_summary: pd.DataFrame,
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
    ali_diagnostics: pd.DataFrame,
    periods_per_year: float,
    md_threshold: float,
    disagreement_threshold: float,
) -> dict[str, Any]:
    selected_metrics = calculate_strategy_metrics(selected_returns, periods_per_year).to_dict()
    top_ic = (
        ic_fdr.assign(abs_ic=ic_fdr["spearman_ic"].abs())
        .sort_values("abs_ic", ascending=False)
        .head(20)
        .to_dict(orient="records")
    )
    fdr_significant = int((ic_fdr["bh_qvalue"] <= 0.05).sum()) if not ic_fdr.empty else 0
    month_net = (
        selected_returns.assign(
            month=pd.to_datetime(bars["end_ts"], utc=True).dt.to_period("M").astype(str)
        )
        .groupby("month")["net_return"]
        .sum()
    )

    return {
        "roots": list(roots),
        "core_roots": list(core_roots),
        "isolated_roots": list(isolated_roots),
        "start": config.start,
        "end": config.end,
        "trade_inventory": trade_inventory.to_dict(orient="records"),
        "primary_complete_bars": len(bars),
        "periods_per_year": periods_per_year,
        "threshold_summary": threshold_summary.to_dict(orient="records"),
        "mean_core_contribution": {
            root: float(bars[f"{root}_notional"].sum() / bars["bar_notional"].sum())
            for root in core_roots
        },
        "md_threshold": md_threshold,
        "large_small_l1_threshold": disagreement_threshold,
        "filter_pass_rate": float(
            (
                (anomalies["md_rolling"] >= md_threshold)
                & (size_disagreement["large_small_l1_distance"] >= disagreement_threshold)
            ).mean()
        ),
        "selected_strategy_metrics": selected_metrics,
        "cost_variants": variants.to_dict(orient="records"),
        "cost_estimates": cost_estimates.to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "split_metrics": split_metrics.to_dict(orient="records"),
        "monthly_net_positive_fraction": float((month_net > 0).mean()),
        "top_information_coefficients": top_ic,
        "fdr_significant_tests": fdr_significant,
        "monthly_ic_summary": monthly_ic.groupby("root")["spearman_ic"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .to_dict(orient="records")
        if not monthly_ic.empty
        else [],
        "ali_diagnostics": ali_diagnostics.to_dict(orient="records"),
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
    }


def _periods_per_year(bars: pd.DataFrame) -> float:
    elapsed_years = (
        pd.to_datetime(bars["end_ts"], utc=True).max()
        - pd.to_datetime(bars["start_ts"], utc=True).min()
    ).total_seconds() / (365.25 * 86_400.0)
    return len(bars) / elapsed_years if elapsed_years > 0 else float(len(bars))


if __name__ == "__main__":
    main()
