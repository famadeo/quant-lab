from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TRADE_REQUIRED_COLUMNS = frozenset({"ts_event", "symbol", "price", "size", "side"})
CONTINUOUS_REQUIRED_COLUMNS = frozenset({"ts", "active", "cont_close", "cont_logprice", "is_roll"})
ALLOWED_AGGRESSOR_SIDES = frozenset({"A", "B", "N"})


@dataclass(frozen=True)
class TradeQualitySummary:
    root: str
    rows: int
    start_ts: pd.Timestamp | None
    end_ts: pd.Timestamp | None
    symbol_count: int
    trade_count: int
    duplicate_rows: int
    notional: float
    buy_notional_share: float
    sell_notional_share: float
    neutral_notional_share: float
    min_price: float
    max_price: float
    min_size: float
    max_size: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "rows": self.rows,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "symbol_count": self.symbol_count,
            "trade_count": self.trade_count,
            "duplicate_rows": self.duplicate_rows,
            "notional": self.notional,
            "buy_notional_share": self.buy_notional_share,
            "sell_notional_share": self.sell_notional_share,
            "neutral_notional_share": self.neutral_notional_share,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "min_size": self.min_size,
            "max_size": self.max_size,
        }


@dataclass(frozen=True)
class ContinuousQualitySummary:
    root: str
    rows: int
    start_ts: pd.Timestamp | None
    end_ts: pd.Timestamp | None
    active_contracts: int
    active_switches: int
    roll_rows: int
    duplicate_timestamps: int
    missing_close_rows: int
    min_close: float
    max_close: float
    max_abs_log_return: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "rows": self.rows,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "active_contracts": self.active_contracts,
            "active_switches": self.active_switches,
            "roll_rows": self.roll_rows,
            "duplicate_timestamps": self.duplicate_timestamps,
            "missing_close_rows": self.missing_close_rows,
            "min_close": self.min_close,
            "max_close": self.max_close,
            "max_abs_log_return": self.max_abs_log_return,
        }


def validate_trade_frame(frame: pd.DataFrame, *, root: str | None = None) -> None:
    missing = sorted(TRADE_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        label = f" for {root}" if root else ""
        raise ValueError(f"trade frame{label} is missing required columns: {missing}")

    if frame.empty:
        return

    price = pd.to_numeric(frame["price"], errors="coerce")
    size = pd.to_numeric(frame["size"], errors="coerce")
    if price.isna().any() or (price <= 0.0).any():
        raise ValueError(f"trade frame has non-positive or missing price values: {root}")
    if size.isna().any() or (size <= 0.0).any():
        raise ValueError(f"trade frame has non-positive or missing size values: {root}")
    if frame["symbol"].isna().any():
        raise ValueError(f"trade frame has missing symbols: {root}")

    sides = frame["side"].astype("string")
    invalid_sides = sorted(set(sides.dropna().astype(str)) - ALLOWED_AGGRESSOR_SIDES)
    if invalid_sides or sides.isna().any():
        raise ValueError(f"trade frame has invalid aggressor sides for {root}: {invalid_sides}")


def summarize_trade_frame(
    frame: pd.DataFrame,
    *,
    root: str,
    multiplier: float,
) -> TradeQualitySummary:
    validate_trade_frame(frame, root=root)
    if frame.empty:
        return TradeQualitySummary(
            root=root,
            rows=0,
            start_ts=None,
            end_ts=None,
            symbol_count=0,
            trade_count=0,
            duplicate_rows=0,
            notional=0.0,
            buy_notional_share=np.nan,
            sell_notional_share=np.nan,
            neutral_notional_share=np.nan,
            min_price=np.nan,
            max_price=np.nan,
            min_size=np.nan,
            max_size=np.nan,
        )

    ts = pd.to_datetime(frame["ts_event"], utc=True)
    price = frame["price"].astype(float)
    size = frame["size"].astype(float)
    notional = price * size * float(multiplier)
    total_notional = float(notional.sum())
    by_side = notional.groupby(frame["side"].astype("string")).sum()

    def side_share(side: str) -> float:
        if total_notional <= 0.0:
            return np.nan
        return float(by_side.get(side, 0.0) / total_notional)

    duplicate_subset = ["ts_event", "symbol", "price", "size", "side"]
    return TradeQualitySummary(
        root=root,
        rows=len(frame),
        start_ts=ts.min(),
        end_ts=ts.max(),
        symbol_count=int(frame["symbol"].nunique(dropna=True)),
        trade_count=len(frame),
        duplicate_rows=int(frame.duplicated(subset=duplicate_subset).sum()),
        notional=total_notional,
        buy_notional_share=side_share("B"),
        sell_notional_share=side_share("A"),
        neutral_notional_share=side_share("N"),
        min_price=float(price.min()),
        max_price=float(price.max()),
        min_size=float(size.min()),
        max_size=float(size.max()),
    )


def summarize_trade_file(
    path: Path,
    *,
    root: str,
    multiplier: float,
) -> TradeQualitySummary:
    return summarize_trade_frame(pd.read_parquet(path), root=root, multiplier=multiplier)


def validate_continuous_frame(frame: pd.DataFrame, *, root: str | None = None) -> None:
    missing = sorted(CONTINUOUS_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        label = f" for {root}" if root else ""
        raise ValueError(f"continuous frame{label} is missing required columns: {missing}")
    if frame.empty:
        return
    close = pd.to_numeric(frame["cont_close"], errors="coerce")
    if close.isna().any() or (close <= 0.0).any():
        raise ValueError(f"continuous frame has non-positive or missing closes: {root}")
    if frame["active"].isna().any():
        raise ValueError(f"continuous frame has missing active contracts: {root}")


def summarize_continuous_frame(
    frame: pd.DataFrame,
    *,
    root: str,
) -> ContinuousQualitySummary:
    validate_continuous_frame(frame, root=root)
    if frame.empty:
        return ContinuousQualitySummary(
            root=root,
            rows=0,
            start_ts=None,
            end_ts=None,
            active_contracts=0,
            active_switches=0,
            roll_rows=0,
            duplicate_timestamps=0,
            missing_close_rows=0,
            min_close=np.nan,
            max_close=np.nan,
            max_abs_log_return=np.nan,
        )

    ts = pd.to_datetime(frame["ts"], utc=True)
    active = frame["active"].astype("string")
    close = frame["cont_close"].astype(float)
    log_returns = np.log(close).diff().replace([np.inf, -np.inf], np.nan)
    active_switches = active.ne(active.shift()).fillna(False)
    active_switches.iloc[0] = False
    is_roll = frame["is_roll"].fillna(False).astype(bool)
    return ContinuousQualitySummary(
        root=root,
        rows=len(frame),
        start_ts=ts.min(),
        end_ts=ts.max(),
        active_contracts=int(active.nunique(dropna=True)),
        active_switches=int(active_switches.sum()),
        roll_rows=int(is_roll.sum()),
        duplicate_timestamps=int(ts.duplicated().sum()),
        missing_close_rows=int(frame["cont_close"].isna().sum()),
        min_close=float(close.min()),
        max_close=float(close.max()),
        max_abs_log_return=float(log_returns.abs().max()),
    )


def summarize_continuous_file(path: Path, *, root: str) -> ContinuousQualitySummary:
    return summarize_continuous_frame(pd.read_parquet(path), root=root)


def align_continuous_marks_to_bars(
    continuous_by_root: dict[str, pd.DataFrame],
    bars: pd.DataFrame,
    roots: tuple[str, ...],
    *,
    max_staleness_seconds: float,
    roll_cooldown_bars: int,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Sample roll-adjusted continuous marks as-of each bar end and mask unsafe bars.

    Marks are taken with ``merge_asof(direction="backward")`` so only information at or
    before each bar end is used (no price lookahead). A bar is invalidated when a mark is
    stale (older than ``max_staleness_seconds``) or within ``roll_cooldown_bars`` of an
    active-contract switch.

    The roll-cooldown window is deliberately symmetric (it also masks the ``roll_cooldown``
    bars *before* a switch). This is not price lookahead: futures roll dates come from the
    known expiry calendar, so an imminent switch is knowable ex-ante. The mask only ever
    *removes* bars from the tradable set, so it cannot manufacture PnL; its sole purpose is
    to drop returns spanning a contract change. Keep it symmetric on purpose.
    """
    if "end_ts" not in bars:
        raise ValueError("bars must contain an end_ts column")

    end_ts = pd.to_datetime(bars["end_ts"], utc=True).rename("end_ts")
    query = pd.DataFrame({"end_ts": end_ts})
    log_prices = pd.DataFrame(index=bars.index)
    continuous_marks = pd.DataFrame(index=bars.index)
    price_validity = pd.DataFrame(index=bars.index)

    for root in roots:
        if root not in continuous_by_root:
            raise ValueError(f"missing continuous frame for {root}")
        frame = continuous_by_root[root].copy()
        validate_continuous_frame(frame, root=root)
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
        frame = frame.sort_values("ts")
        frame = frame.loc[
            (frame["ts"] >= end_ts.min() - pd.Timedelta(days=2))
            & (frame["ts"] <= end_ts.max() + pd.Timedelta(minutes=1))
        ].copy()
        if frame.empty:
            raise ValueError(f"continuous frame for {root} has no rows overlapping bar timestamps")

        frame["log_price"] = np.log(frame["cont_close"].astype(float).iloc[0]) + frame[
            "cont_logprice"
        ].astype(float)
        aligned = pd.merge_asof(
            query,
            frame[["ts", "active", "cont_close", "log_price", "is_roll"]],
            left_on="end_ts",
            right_on="ts",
            direction="backward",
        )
        staleness = (aligned["end_ts"] - aligned["ts"]).dt.total_seconds()
        active = aligned["active"].astype("string")
        active_switch = active.ne(active.shift()).fillna(False)
        active_switch.iloc[0] = False
        stale = staleness > max_staleness_seconds

        log_prices[root] = aligned["log_price"].to_numpy()
        continuous_marks[f"{root}_active"] = active.to_numpy()
        continuous_marks[f"{root}_continuous_close"] = aligned["cont_close"].to_numpy()
        continuous_marks[f"{root}_continuous_ts"] = aligned["ts"].to_numpy()
        continuous_marks[f"{root}_staleness_seconds"] = staleness.to_numpy()
        continuous_marks[f"{root}_active_switch"] = active_switch.to_numpy()
        continuous_marks[f"{root}_source_is_roll"] = np.where(
            aligned["is_roll"].isna(),
            False,
            aligned["is_roll"],
        ).astype(bool)
        price_validity[f"{root}_fresh"] = (~stale).fillna(False).to_numpy()
        price_validity[f"{root}_active_switch"] = active_switch.to_numpy()

    roll_any = price_validity[[f"{root}_active_switch" for root in roots]].any(axis=1)
    roll_invalid = roll_any.copy()
    for offset in range(1, roll_cooldown_bars + 1):
        roll_invalid = (
            roll_invalid
            | roll_any.shift(offset, fill_value=False).astype(bool)
            | roll_any.shift(-offset, fill_value=False).astype(bool)
        )
    fresh_all = price_validity[[f"{root}_fresh" for root in roots]].all(axis=1)
    valid_price_mask = fresh_all & ~roll_invalid
    price_validity["fresh_all"] = fresh_all
    price_validity["roll_any"] = roll_any
    price_validity["roll_invalid"] = roll_invalid
    price_validity["valid_price_mask"] = valid_price_mask
    log_prices = log_prices.where(valid_price_mask, axis=0)
    return {
        "log_prices": log_prices,
        "continuous_marks": continuous_marks,
        "price_validity": price_validity,
        "valid_price_mask": valid_price_mask,
    }


def bar_log_returns_with_validity(
    log_prices: pd.DataFrame,
    valid_price_mask: pd.Series,
) -> pd.DataFrame:
    returns = log_prices.astype(float).diff()
    valid_return = valid_price_mask & valid_price_mask.shift(1, fill_value=False).astype(bool)
    return returns.where(valid_return, axis=0)
