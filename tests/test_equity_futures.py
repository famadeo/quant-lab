from __future__ import annotations

import pandas as pd

from quantlab.data.equity_futures import build_continuous_equity_futures_root


def test_build_continuous_equity_futures_root_rolls_by_forward_volume() -> None:
    raw = pd.DataFrame(
        {
            "ts_event": pd.to_datetime(
                [
                    "2026-01-02T14:30:00Z",
                    "2026-01-02T14:31:00Z",
                    "2026-01-02T14:30:00Z",
                    "2026-01-02T14:31:00Z",
                    "2026-01-05T14:30:00Z",
                    "2026-01-05T14:31:00Z",
                    "2026-01-05T14:30:00Z",
                    "2026-01-05T14:31:00Z",
                ]
            ),
            "symbol": ["AAH6", "AAH6", "AAM6", "AAM6", "AAH6", "AAH6", "AAM6", "AAM6"],
            "close": [100.0, 101.0, 98.0, 99.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [100.0, 100.0, 20.0, 20.0, 10.0, 10.0, 200.0, 200.0],
        }
    )

    continuous = build_continuous_equity_futures_root(raw, "AA")

    assert continuous["active"].tolist() == ["AAH6", "AAH6", "AAM6", "AAM6"]
    assert continuous["is_roll"].tolist() == [False, False, True, False]
    assert bool(continuous["cont_logprice"].notna().all())
