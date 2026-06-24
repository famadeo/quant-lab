from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quantlab.metals_flow.config import MetalsFlowConfig

TRADE_COLUMNS = ["ts_event", "price", "size", "side", "root", "notional", "signed_notional"]


def load_or_build_trade_cache(config: MetalsFlowConfig) -> pd.DataFrame:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.cache_dir / f"trades_{config.date_tag}.parquet"
    if cache_path.exists():
        return _normalize_trade_frame(pd.read_parquet(cache_path))

    frames: list[pd.DataFrame] = []
    start = pd.Timestamp(config.start)
    end = pd.Timestamp(config.end)
    for root in config.roots:
        path = config.trade_dir / f"{root}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=["ts_event", "price", "size", "side"])
        frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
        frame = frame[(frame["ts_event"] >= start) & (frame["ts_event"] < end)].copy()
        frame["root"] = root
        frame["notional"] = (
            frame["price"].astype(float) * frame["size"].astype(float) * config.multipliers[root]
        )
        sign = frame["side"].map({"B": 1.0, "A": -1.0}).fillna(0.0)
        frame["signed_notional"] = frame["notional"] * sign
        frames.append(frame.loc[:, TRADE_COLUMNS])

    trades = pd.concat(frames, ignore_index=True)
    trades = _normalize_trade_frame(trades)
    trades.to_parquet(cache_path, index=False)
    return trades


def load_or_build_bars(
    config: MetalsFlowConfig,
    trades: pd.DataFrame,
    thresholds: Iterable[float] | None = None,
) -> dict[float, pd.DataFrame]:
    bars_by_threshold: dict[float, pd.DataFrame] = {}
    for threshold_value in thresholds or config.thresholds:
        threshold = float(threshold_value)
        path = config.cache_dir / f"bars_{int(threshold)}_{config.date_tag}.parquet"
        if path.exists():
            bars = pd.read_parquet(path)
            bars["start_ts"] = pd.to_datetime(bars["start_ts"], utc=True)
            bars["end_ts"] = pd.to_datetime(bars["end_ts"], utc=True)
        else:
            bars = build_cross_sectional_dollar_bars_fast(trades, config.roots, threshold)
            bars.to_parquet(path, index=False)
        bars_by_threshold[threshold] = bars
    return bars_by_threshold


def build_cross_sectional_dollar_bars_fast(
    trades: pd.DataFrame,
    roots: tuple[str, ...],
    threshold: float,
) -> pd.DataFrame:
    trades = _normalize_trade_frame(trades)
    if trades.empty:
        return pd.DataFrame()

    root_to_code = {root: index for index, root in enumerate(roots)}
    root_codes = trades["root"].map(root_to_code).to_numpy()
    keep = pd.notna(root_codes)
    frame = trades.loc[keep].reset_index(drop=True)
    root_codes = frame["root"].map(root_to_code).to_numpy(dtype=np.int64)
    notionals = frame["notional"].to_numpy(dtype=float)
    signed = frame["signed_notional"].to_numpy(dtype=float)
    timestamps = frame["ts_event"].to_numpy()
    cumulative = np.cumsum(notionals)

    rows: list[dict[str, object]] = []
    start = 0
    base = 0.0
    bar_id = 0
    n_roots = len(roots)
    n_trades = len(frame)

    while start < n_trades:
        target = base + threshold
        end = int(np.searchsorted(cumulative, target, side="left"))
        complete = end < n_trades
        if not complete:
            end = n_trades - 1

        root_slice = root_codes[start : end + 1]
        notional_slice = notionals[start : end + 1]
        signed_slice = signed[start : end + 1]
        bar_notional = float(notional_slice.sum())
        counts_array = np.bincount(root_slice, minlength=n_roots)
        notional_array = np.bincount(root_slice, weights=notional_slice, minlength=n_roots)
        signed_array = np.bincount(root_slice, weights=signed_slice, minlength=n_roots)

        rows.append(
            _bar_record(
                bar_id=bar_id,
                start_ts=pd.Timestamp(timestamps[start]),
                end_ts=pd.Timestamp(timestamps[end]),
                threshold=threshold,
                bar_notional=bar_notional,
                trade_count=int(end - start + 1),
                complete=complete,
                roots=roots,
                notional=dict(zip(roots, notional_array, strict=True)),
                signed_notional=dict(zip(roots, signed_array, strict=True)),
                counts=dict(zip(roots, counts_array.astype(int), strict=True)),
            )
        )
        if not complete:
            break
        base = float(cumulative[end])
        start = end + 1
        bar_id += 1

    return pd.DataFrame(rows)


def build_cross_sectional_dollar_bars(
    trades: pd.DataFrame,
    roots: tuple[str, ...],
    threshold: float,
) -> pd.DataFrame:
    trades = _normalize_trade_frame(trades)
    rows: list[dict[str, object]] = []

    bar_id = 0
    start_ts: pd.Timestamp | None = None
    bar_notional = 0.0
    trade_count = 0
    notional = {root: 0.0 for root in roots}
    signed_notional = {root: 0.0 for root in roots}
    counts = {root: 0 for root in roots}

    for row in trades.itertuples(index=False):
        ts = row.ts_event
        root = row.root
        if root not in notional:
            continue
        if start_ts is None:
            start_ts = ts

        value = float(row.notional)
        signed_value = float(row.signed_notional)
        bar_notional += value
        trade_count += 1
        notional[root] += value
        signed_notional[root] += signed_value
        counts[root] += 1

        if bar_notional >= threshold:
            rows.append(
                _bar_record(
                    bar_id=bar_id,
                    start_ts=start_ts,
                    end_ts=ts,
                    threshold=threshold,
                    bar_notional=bar_notional,
                    trade_count=trade_count,
                    complete=True,
                    roots=roots,
                    notional=notional,
                    signed_notional=signed_notional,
                    counts=counts,
                )
            )
            bar_id += 1
            start_ts = None
            bar_notional = 0.0
            trade_count = 0
            notional = {root: 0.0 for root in roots}
            signed_notional = {root: 0.0 for root in roots}
            counts = {root: 0 for root in roots}

    if start_ts is not None and trade_count:
        rows.append(
            _bar_record(
                bar_id=bar_id,
                start_ts=start_ts,
                end_ts=trades["ts_event"].iloc[-1],
                threshold=threshold,
                bar_notional=bar_notional,
                trade_count=trade_count,
                complete=False,
                roots=roots,
                notional=notional,
                signed_notional=signed_notional,
                counts=counts,
            )
        )

    return pd.DataFrame(rows)


def endpoint_prices(
    trades: pd.DataFrame,
    bars: pd.DataFrame,
    roots: tuple[str, ...],
) -> pd.DataFrame:
    trades = _normalize_trade_frame(trades)
    end_ts = pd.to_datetime(bars["end_ts"], utc=True)
    end_ns = end_ts.astype("int64").to_numpy()
    prices: dict[str, np.ndarray] = {}

    for root in roots:
        root_trades = trades.loc[trades["root"] == root, ["ts_event", "price"]].sort_values(
            "ts_event"
        )
        if root_trades.empty:
            prices[root] = np.full(len(end_ts), np.nan)
            continue
        ts_values = root_trades["ts_event"].astype("int64").to_numpy()
        price_values = root_trades["price"].astype(float).to_numpy()
        indexer = np.searchsorted(ts_values, end_ns, side="right") - 1
        root_prices = np.where(indexer >= 0, price_values[np.maximum(indexer, 0)], np.nan)
        prices[root] = root_prices

    return pd.DataFrame(prices, index=end_ts).rename_axis("end_ts")


def bar_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices.astype(float)).diff()
    return returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def assign_trades_to_bars(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.Series:
    trades = _normalize_trade_frame(trades)
    end_ns = pd.to_datetime(bars["end_ts"], utc=True).astype("int64").to_numpy()
    trade_ns = trades["ts_event"].astype("int64").to_numpy()
    positions = np.searchsorted(end_ns, trade_ns, side="left")
    assigned = pd.Series(positions, index=trades.index, name="bar_id")
    assigned = assigned.where(assigned < len(bars), other=pd.NA).astype("Int64")
    return assigned


def summarize_bars(bars_by_threshold: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for threshold, bars in sorted(bars_by_threshold.items()):
        complete = bars[bars["complete"]].copy()
        if complete.empty:
            continue
        elapsed_days = (
            complete["end_ts"].max() - complete["start_ts"].min()
        ).total_seconds() / 86_400.0
        rows.append(
            {
                "threshold": threshold,
                "threshold_m": threshold / 1_000_000.0,
                "bars": len(complete),
                "bars_per_day": len(complete) / elapsed_days if elapsed_days > 0 else np.nan,
                "median_duration_seconds": complete["duration_seconds"].median(),
                "mean_duration_seconds": complete["duration_seconds"].mean(),
                "median_trades": complete["trades"].median(),
                "mean_trades": complete["trades"].mean(),
                "median_dominant_share": complete["dominant_share"].median(),
                "mean_hhi": complete["hhi_notional_share"].mean(),
            }
        )
    return pd.DataFrame(rows)


def _bar_record(
    *,
    bar_id: int,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    threshold: float,
    bar_notional: float,
    trade_count: int,
    complete: bool,
    roots: tuple[str, ...],
    notional: dict[str, float],
    signed_notional: dict[str, float],
    counts: dict[str, int],
) -> dict[str, object]:
    record: dict[str, object] = {
        "bar_id": bar_id,
        "start_ts": start_ts,
        "end_ts": end_ts,
        "bar_notional": bar_notional,
        "trades": trade_count,
        "complete": complete,
    }
    for root in roots:
        record[f"{root}_notional"] = notional[root]
    for root in roots:
        record[f"{root}_signed_notional"] = signed_notional[root]
    for root in roots:
        record[f"{root}_trades"] = counts[root]

    duration = max((end_ts - start_ts).total_seconds(), 0.0)
    shares = np.array(
        [notional[root] / bar_notional if bar_notional > 0 else 0.0 for root in roots]
    )
    trade_shares = np.array([counts[root] / trade_count if trade_count else 0.0 for root in roots])
    dominant_idx = int(np.argmax(shares)) if len(shares) else 0
    signed_total = sum(signed_notional.values())

    record.update(
        {
            "duration_seconds": duration,
            "overshoot_pct": (bar_notional / threshold) - 1.0 if threshold > 0 else np.nan,
            "threshold": threshold,
            "dominant_root": roots[dominant_idx],
            "dominant_share": float(shares[dominant_idx]),
            "hhi_notional_share": float(np.square(shares).sum()),
            "hhi_trade_share": float(np.square(trade_shares).sum()),
            "complex_signed_notional_ratio": signed_total / bar_notional
            if bar_notional > 0
            else np.nan,
        }
    )
    return record


def _normalize_trade_frame(trades: pd.DataFrame) -> pd.DataFrame:
    frame = trades.loc[:, [column for column in TRADE_COLUMNS if column in trades.columns]].copy()
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
    frame["price"] = frame["price"].astype(float)
    frame["size"] = frame["size"].astype(float)
    frame["notional"] = frame["notional"].astype(float)
    frame["signed_notional"] = frame["signed_notional"].astype(float)
    return frame.sort_values("ts_event", kind="mergesort").reset_index(drop=True)
