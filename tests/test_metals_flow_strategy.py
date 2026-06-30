from __future__ import annotations

import pandas as pd

from quantlab.metals_flow.strategy import residual_momentum_positions


def test_residual_momentum_positions_hold_until_decay_or_timeout() -> None:
    residual_z = pd.DataFrame({"PA": [2.2, 2.4, 1.1, 0.6, 2.3, 2.1, 2.0]})
    entry = pd.Series([True] * len(residual_z))

    positions = residual_momentum_positions(
        residual_z,
        entry,
        entry_z=2.0,
        exit_z=0.75,
        max_holding_bars=3,
        stop_z=None,
    )

    assert positions["PA"].tolist() == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]


def test_residual_momentum_positions_exit_on_sign_reversal_and_cooldown() -> None:
    residual_z = pd.DataFrame({"PA": [2.2, 1.0, -2.4, -2.5, -2.6, -2.7]})
    entry = pd.Series([True] * len(residual_z))

    positions = residual_momentum_positions(
        residual_z,
        entry,
        entry_z=2.0,
        exit_z=0.75,
        max_holding_bars=10,
        stop_z=None,
        cooldown_bars=1,
    )

    assert positions["PA"].tolist() == [1.0, 1.0, 0.0, -1.0, -1.0, -1.0]


def test_residual_momentum_positions_honors_root_entry_masks() -> None:
    residual_z = pd.DataFrame({"PL": [2.5, 2.6], "PA": [2.5, 2.6]})
    entry = pd.Series([True, True])
    root_masks = pd.DataFrame({"PL": [False, False], "PA": [True, True]})

    positions = residual_momentum_positions(
        residual_z,
        entry,
        root_masks,
        entry_z=2.0,
        exit_z=0.75,
        max_holding_bars=10,
        stop_z=None,
    )

    assert positions["PL"].tolist() == [0.0, 0.0]
    assert positions["PA"].tolist() == [1.0, 1.0]
