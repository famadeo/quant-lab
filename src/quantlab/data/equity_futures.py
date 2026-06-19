from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

MONTH_CODES = "FGHJKMNQUVXZ"


@dataclass(frozen=True)
class ContinuousRootSummary:
    root: str
    input_path: str
    output_path: str
    rows: int
    contracts: int
    start: str
    end: str
    rolls: int


def build_continuous_equity_futures_root(raw_bars: pd.DataFrame, root: str) -> pd.DataFrame:
    """Build a return-spliced continuous 1-minute series from outright futures bars."""
    bars = _normalize_raw_bars(raw_bars, root)
    if bars.empty:
        return _empty_continuous_frame()

    schedule = _active_contract_schedule(bars)
    work = bars.merge(schedule, on="session_date", how="left")
    continuous = work.loc[work["symbol"].eq(work["active"])].copy()
    if continuous.empty:
        return _empty_continuous_frame()

    continuous = continuous.sort_values("ts")
    continuous["is_roll"] = continuous["active"].ne(continuous["active"].shift()).fillna(False)
    continuous.loc[continuous["active"].shift().isna(), "is_roll"] = False
    continuous["cont_logret"] = continuous["contract_logret"].fillna(0.0)
    continuous["cont_logprice"] = continuous["cont_logret"].cumsum()
    first_close = float(continuous["close"].iloc[0])
    continuous["cont_close"] = first_close * np.exp(continuous["cont_logprice"])

    return continuous.loc[
        :,
        ["ts", "active", "cont_logret", "cont_close", "volume", "is_roll", "cont_logprice"],
    ].reset_index(drop=True)


def prepare_continuous_equity_futures(
    raw_dir: Path,
    output_dir: Path,
    roots: list[str] | None = None,
) -> list[ContinuousRootSummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    root_paths = _root_paths(raw_dir, roots)
    summaries: list[ContinuousRootSummary] = []

    for root, input_path in root_paths:
        raw = pd.read_parquet(input_path)
        continuous = build_continuous_equity_futures_root(raw, root)
        output_path = output_dir / f"{root}.parquet"
        continuous.to_parquet(output_path, index=False)

        summaries.append(
            ContinuousRootSummary(
                root=root,
                input_path=str(input_path),
                output_path=str(output_path),
                rows=len(continuous),
                contracts=int(continuous["active"].nunique()) if not continuous.empty else 0,
                start=str(continuous["ts"].min()) if not continuous.empty else "",
                end=str(continuous["ts"].max()) if not continuous.empty else "",
                rolls=int(continuous["is_roll"].sum()) if not continuous.empty else 0,
            )
        )

    _write_manifest(output_dir, summaries)
    return summaries


def _normalize_raw_bars(raw_bars: pd.DataFrame, root: str) -> pd.DataFrame:
    required = {"ts_event", "symbol", "close", "volume"}
    missing = required.difference(raw_bars.columns)
    if missing:
        raise ValueError(f"{root} raw bars are missing required columns: {sorted(missing)}")

    symbol_pattern = re.compile(rf"^{re.escape(root)}[{MONTH_CODES}]\d{{1,2}}$")
    bars = raw_bars.copy()
    bars["symbol"] = bars["symbol"].astype(str)
    bars = bars.loc[bars["symbol"].str.match(symbol_pattern)].copy()
    if bars.empty:
        return pd.DataFrame()

    bars["ts"] = pd.to_datetime(bars["ts_event"], utc=True)
    close = cast(pd.Series, pd.to_numeric(bars["close"], errors="coerce"))
    volume = cast(pd.Series, pd.to_numeric(bars["volume"], errors="coerce"))
    bars["close"] = close
    bars["volume"] = volume.fillna(0.0).astype(float)
    bars = bars.loc[bars["ts"].notna() & bars["close"].gt(0)].copy()
    if bars.empty:
        return pd.DataFrame()

    bars = bars.sort_values(["symbol", "ts"])
    bars["session_date"] = bars["ts"].dt.tz_convert("America/New_York").dt.date
    bars["log_close"] = np.log(bars["close"])
    bars["contract_logret"] = bars.groupby("symbol", sort=False)["log_close"].diff()
    return bars


def _active_contract_schedule(bars: pd.DataFrame) -> pd.DataFrame:
    daily_volume = cast(
        pd.DataFrame,
        bars.groupby(["session_date", "symbol"], as_index=False).agg(volume=("volume", "sum")),
    ).sort_values(by=["session_date"])
    last_timestamp = cast(
        pd.DataFrame,
        bars.loc[:, ["symbol", "ts"]].groupby("symbol", as_index=False).max(),
    ).sort_values(by=["ts"])
    expiry_rank: dict[str, int] = {
        str(symbol): index
        for index, (symbol, _ts) in enumerate(
            last_timestamp.loc[:, ["symbol", "ts"]].itertuples(index=False, name=None)
        )
    }

    active_by_date: list[dict[str, Any]] = []
    current: str | None = None
    for session_date, group in daily_volume.groupby("session_date", sort=True):
        volume_by_symbol: dict[str, float] = {}
        for symbol, volume_value in group.loc[:, ["symbol", "volume"]].itertuples(
            index=False, name=None
        ):
            numeric_volume = float(volume_value)
            if numeric_volume > 0:
                volume_by_symbol[str(symbol)] = numeric_volume
        if not volume_by_symbol:
            active_by_date.append({"session_date": session_date, "active": current})
            continue
        if current is None or current not in volume_by_symbol:
            current = max(volume_by_symbol, key=lambda symbol: volume_by_symbol[symbol])
        else:
            later_symbols = [
                symbol
                for symbol in volume_by_symbol
                if expiry_rank[symbol] > expiry_rank.get(current, -1)
            ]
            if later_symbols:
                best_later = max(later_symbols, key=lambda symbol: volume_by_symbol[symbol])
                if volume_by_symbol[best_later] > volume_by_symbol[current]:
                    current = best_later
        active_by_date.append({"session_date": session_date, "active": current})

    return pd.DataFrame(active_by_date)


def _root_paths(raw_dir: Path, roots: list[str] | None) -> list[tuple[str, Path]]:
    if roots:
        return [(root, raw_dir / f"{root}.parquet") for root in roots]
    return [(path.stem, path) for path in sorted(raw_dir.glob("*.parquet"))]


def _empty_continuous_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.Series(dtype="datetime64[ns, UTC]"),
            "active": pd.Series(dtype="string"),
            "cont_logret": pd.Series(dtype="float64"),
            "cont_close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "is_roll": pd.Series(dtype="bool"),
            "cont_logprice": pd.Series(dtype="float64"),
        }
    )


def _write_manifest(output_dir: Path, summaries: list[ContinuousRootSummary]) -> None:
    rows = [asdict(summary) for summary in summaries]
    pd.DataFrame(rows).to_csv(output_dir / "manifest.csv", index=False)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "root_count": len(summaries),
        "roots": [summary.root for summary in summaries],
        "total_rows": sum(summary.rows for summary in summaries),
        "summaries": rows,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
