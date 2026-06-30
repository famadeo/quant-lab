from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

EXP = pathlib.Path(__file__).resolve().parents[1] / "experiments"
SOURCES = {
    "sync": EXP / "HYP-0031-metals-convenience-yield-basis-5m-sync",
    "daily": EXP / "HYP-0030-metals-convenience-yield-basis",
}
OUT = EXP / "HYP-0053-metals-convenience-yield-mechanism"
VARIANTS = ["target3m_minv10", "target1m_minv500", "target1m_minv1000"]
ROOTS = ["GC", "SI", "HG", "PL", "PA"]
MIN_Z_OBSERVATIONS = 40  # matches run_metals_convenience_yield_basis_backtest.py


def reconstruct_z(panel, target_months, min_volume, lookback):
    data = panel[(panel["target_months"] == target_months)
                 & (panel["min_volume"] == min_volume)].copy()
    data = data.sort_values(["root", "date"])

    def _z(g):
        roll = g["carry_pct_ann"].shift(1).rolling(lookback, min_periods=MIN_Z_OBSERVATIONS)
        g["carry_mean_lagged"] = roll.mean()
        g["carry_std_lagged"] = roll.std(ddof=0)
        g["carry_z"] = (g["carry_pct_ann"] - g["carry_mean_lagged"]) / g["carry_std_lagged"]
        g.loc[g["carry_std_lagged"] <= 0, "carry_z"] = np.nan
        return g

    result = data.groupby("root", group_keys=False).apply(_z, include_groups=False)
    result["root"] = data["root"]  # restore groupby key dropped by include_groups=False
    return result


def _variant_params(strategy_metrics, variant_name):
    row = strategy_metrics[strategy_metrics["variant"].str.startswith(variant_name + "_")]
    row = row[row["cost_multiplier"] == 1.0].iloc[0]
    return {k: row[k] for k in ["target_months", "min_volume", "lookback", "entry_z", "exit_z"]}


def load_variant(source_dir, variant_name):
    panel = pd.read_parquet(source_dir / "curve_panel.parquet")
    sm = pd.read_csv(source_dir / "strategy_metrics.csv")
    p = _variant_params(sm, variant_name)
    zpanel = reconstruct_z(
        panel, int(p["target_months"]), float(p["min_volume"]), int(p["lookback"])
    )
    events = pd.read_csv(source_dir / "event_log.csv", parse_dates=["entry_date", "exit_date"])
    lb_tag = f"_lb{int(p['lookback'])}_"
    events = events[events["variant"].str.startswith(variant_name + "_")
                    & events["variant"].str.contains(lb_tag, regex=False)
                    & (events["cost_multiplier"] == 1.0)].copy()
    zpanel["date"] = pd.to_datetime(zpanel["date"])
    return zpanel, events, p
