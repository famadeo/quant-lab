from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import databento as db
import pandas as pd
from dotenv import load_dotenv


@dataclass(frozen=True)
class DatabentoEquityUniverseConfig:
    top_n: int = 100
    dataset: str = "EQUS.MINI"
    schema: str = "ohlcv-1d"
    countries: tuple[str, ...] = ("US",)
    security_types: tuple[str, ...] = ("EQS",)
    as_of: date | None = None
    price_lookback_days: int = 7
    symbols_per_request: int = 1_000
    min_price: float = 1.0
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.price_lookback_days <= 0:
            raise ValueError("price_lookback_days must be positive")
        if self.symbols_per_request <= 0:
            raise ValueError("symbols_per_request must be positive")
        if self.min_price < 0:
            raise ValueError("min_price must be non-negative")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive when provided")


def prepare_security_master_universe(security_master: pd.DataFrame) -> pd.DataFrame:
    """Filter Databento security-master rows to liquid US equity candidates."""
    required = {"nasdaq_symbol", "shares_outstanding"}
    missing = required.difference(security_master.columns)
    if missing:
        raise ValueError(f"security master is missing required columns: {sorted(missing)}")

    frame = security_master.copy()
    frame["symbol"] = frame["nasdaq_symbol"].astype("string").str.strip()
    frame["shares_outstanding"] = pd.to_numeric(frame["shares_outstanding"], errors="coerce")
    mask = frame["symbol"].notna() & frame["shares_outstanding"].gt(0)

    optional_filters = {
        "listing_status": "L",
        "listing_country": "US",
        "trading_currency": "USD",
        "security_type": "EQS",
    }
    for column, value in optional_filters.items():
        if column in frame.columns:
            mask &= frame[column].eq(value)

    frame = frame.loc[mask].copy()
    if frame.empty:
        return frame

    sort_columns = [
        column for column in ["symbol", "shares_outstanding"] if column in frame.columns
    ]
    frame = frame.sort_values(sort_columns, ascending=[True, False])
    frame = frame.drop_duplicates("symbol", keep="first")
    return frame.reset_index(drop=True)


def latest_closes_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return the latest close by symbol from a Databento OHLCV dataframe."""
    frame = bars.copy()
    if "ts_event" not in frame.columns:
        if frame.index.name == "ts_event":
            frame = frame.reset_index()
        elif isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.reset_index(names="ts_event")
        else:
            raise ValueError("bars must include a ts_event column or DatetimeIndex")

    required = {"symbol", "close", "ts_event"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"bars are missing required columns: {sorted(missing)}")

    frame["symbol"] = frame["symbol"].astype("string").str.strip()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
    frame = frame.dropna(subset=["symbol", "close", "ts_event"])
    frame = frame.sort_values(["symbol", "ts_event"])
    latest = frame.groupby("symbol", as_index=False).tail(1)
    return latest.rename(columns={"ts_event": "price_ts"}).loc[:, ["symbol", "price_ts", "close"]]


def rank_by_market_cap(
    security_master: pd.DataFrame,
    closes: pd.DataFrame,
    *,
    top_n: int = 100,
    min_price: float = 1.0,
) -> pd.DataFrame:
    candidates = prepare_security_master_universe(security_master)
    if candidates.empty:
        return candidates

    latest_closes = latest_closes_from_bars(closes)
    merged = candidates.merge(latest_closes, on="symbol", how="inner")
    merged = merged.loc[merged["close"].ge(min_price)].copy()
    merged["market_cap"] = merged["close"] * merged["shares_outstanding"]
    merged = merged.sort_values("market_cap", ascending=False).head(top_n).reset_index(drop=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))

    preferred_columns = [
        "rank",
        "symbol",
        "issuer_name",
        "security_description",
        "primary_exchange",
        "exchange",
        "operating_mic",
        "listing_country",
        "trading_currency",
        "shares_outstanding",
        "shares_outstanding_date",
        "price_ts",
        "close",
        "market_cap",
    ]
    return merged.loc[:, [column for column in preferred_columns if column in merged.columns]]


def build_databento_top_market_cap_universe(
    config: DatabentoEquityUniverseConfig,
    *,
    output_path: Path,
    env_file: Path | None = None,
) -> pd.DataFrame:
    _load_env_file(env_file)
    if not os.getenv("DATABENTO_API_KEY"):
        raise RuntimeError("DATABENTO_API_KEY is not set")

    as_of = config.as_of or date.today()
    start = as_of - timedelta(days=config.price_lookback_days)
    end = as_of + timedelta(days=1)

    reference = db.Reference()
    historical = db.Historical()
    security_master = reference.security_master.get_last(
        countries=list(config.countries),
        security_types=list(config.security_types),
    )
    candidates = prepare_security_master_universe(security_master)
    if config.max_candidates is not None:
        candidates = candidates.nlargest(config.max_candidates, "shares_outstanding")

    symbols = sorted(str(symbol) for symbol in candidates["symbol"].dropna().tolist())
    bars = _fetch_daily_closes(
        historical,
        dataset=config.dataset,
        schema=config.schema,
        symbols=symbols,
        start=start.isoformat(),
        end=end.isoformat(),
        symbols_per_request=config.symbols_per_request,
    )
    universe = rank_by_market_cap(
        candidates,
        bars,
        top_n=config.top_n,
        min_price=config.min_price,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output_path, index=False)
    return universe


def _fetch_daily_closes(
    historical: Any,
    *,
    dataset: str,
    schema: str,
    symbols: list[str],
    start: str,
    end: str,
    symbols_per_request: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in _chunks(symbols, symbols_per_request):
        data = historical.timeseries.get_range(
            dataset=dataset,
            schema=schema,
            symbols=chunk,
            start=start,
            end=end,
        )
        frame = data.to_df(map_symbols=True).reset_index()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(
            {
                "ts_event": pd.Series(dtype="datetime64[ns, UTC]"),
                "symbol": pd.Series(dtype="string"),
                "close": pd.Series(dtype="float64"),
            }
        )
    return pd.concat(frames, ignore_index=True)


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _load_env_file(env_file: Path | None) -> None:
    if env_file is None:
        return
    load_dotenv(env_file)
