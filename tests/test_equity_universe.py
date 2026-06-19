from __future__ import annotations

import pandas as pd

from quantlab.data.equity_universe import latest_closes_from_bars, rank_by_market_cap


def test_latest_closes_from_bars_keeps_latest_symbol_observation() -> None:
    latest_aaa_close = 11.0
    latest_bbb_close = 20.0
    bars = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2026-06-17T00:00:00Z",
                    "2026-06-18T00:00:00Z",
                    "2026-06-17T00:00:00Z",
                ]
            ),
            "symbol": ["AAA", "AAA", "BBB"],
            "close": [10.0, latest_aaa_close, latest_bbb_close],
        }
    )

    closes = latest_closes_from_bars(bars)

    assert closes.set_index("symbol").loc["AAA", "close"] == latest_aaa_close
    assert closes.set_index("symbol").loc["BBB", "close"] == latest_bbb_close


def test_rank_by_market_cap_filters_and_sorts_equities() -> None:
    security_master = pd.DataFrame(
        {
            "nasdaq_symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "issuer_name": ["A", "B", "C", "D", "E"],
            "listing_status": ["L", "L", "D", "L", "L"],
            "listing_country": ["US", "US", "US", "CA", "US"],
            "trading_currency": ["USD", "USD", "USD", "USD", "USD"],
            "security_type": ["EQS", "EQS", "EQS", "EQS", "ETF"],
            "shares_outstanding": [100.0, 50.0, 10_000.0, 10_000.0, 10_000.0],
        }
    )
    bars = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2026-06-18T00:00:00Z",
                    "2026-06-18T00:00:00Z",
                    "2026-06-18T00:00:00Z",
                    "2026-06-18T00:00:00Z",
                    "2026-06-18T00:00:00Z",
                ]
            ),
            "symbol": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "close": [5.0, 20.0, 100.0, 100.0, 100.0],
        }
    )

    ranked = rank_by_market_cap(security_master, bars, top_n=2)

    assert ranked["symbol"].tolist() == ["BBB", "AAA"]
    assert ranked["market_cap"].tolist() == [1_000.0, 500.0]
