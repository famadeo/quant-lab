from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
MONTHS = {
    "F": 1,
    "G": 2,
    "H": 3,
    "J": 4,
    "K": 5,
    "M": 6,
    "N": 7,
    "Q": 8,
    "U": 9,
    "V": 10,
    "X": 11,
    "Z": 12,
}
COLORS = {
    "GC": "#b8860b",
    "SI": "#6f7f8f",
    "HG": "#b15a2a",
    "PL": "#2f7d8c",
    "PA": "#7a4e9b",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path("/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/raw")
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0030-metals-convenience-yield-basis"
COST_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0014-metals-flow-filtered-residual-reversion-3y"
    / "cost_estimates.csv"
)

START = pd.Timestamp("2023-06-22", tz="UTC")
END = pd.Timestamp("2026-06-22", tz="UTC")
MAX_CONTRACT_MONTHS_OUT = 120
MAX_ANCHOR_MONTHS_OUT = 4
BASE_MIN_DAILY_VOLUME = 10.0
MAX_LEG_TS_GAP_MINUTES = 120.0
MIN_Z_OBSERVATIONS = 40
ROOT_WEIGHT = 1.0 / len(ROOTS)
MIN_PERIODS_FOR_ANNUALIZATION = 2

TARGET_MONTHS = [1, 3, 6]
MIN_VOLUME_VARIANTS = [10.0, 500.0, 1_000.0]
LOOKBACKS = [126, 252]
ENTRY_ZS = [1.5, 2.0]
EXIT_ZS = [0.25]
SIDE_MODES = ["both", "backwardation_only", "contango_only"]
COST_MULTIPLIERS = [1.0, 3.0]


@dataclass(frozen=True)
class Variant:
    target_months: int
    min_volume: float
    lookback: int
    entry_z: float
    exit_z: float
    side_mode: str
    cost_multiplier: float

    @property
    def name(self) -> str:
        entry = str(self.entry_z).replace(".", "p")
        exit_value = str(self.exit_z).replace(".", "p")
        min_volume = int(self.min_volume)
        cost = str(self.cost_multiplier).replace(".", "p")
        return (
            f"target{self.target_months}m_minv{min_volume}_lb{self.lookback}_"
            f"entry{entry}_exit{exit_value}_{self.side_mode}_costx{cost}"
        )


def contract_months_out(symbol: str, date_value: object) -> float | None:
    match = re.match(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d)$", str(symbol))
    if not match:
        return None
    month = MONTHS[match.group(2)]
    year_digit = int(match.group(3))
    date = pd.Timestamp(date_value)
    candidates = []
    for year in range(date.year - 1, date.year + 11):
        if year % 10 == year_digit:
            months_out = (year - date.year) * 12 + (month - date.month)
            if 0 <= months_out <= MAX_CONTRACT_MONTHS_OUT:
                candidates.append(months_out)
    return min(candidates) if candidates else None


def load_costs() -> dict[str, float]:
    if not COST_PATH.exists():
        return {"GC": 0.55, "SI": 1.87, "HG": 0.80, "PL": 2.56, "PA": 5.59}
    costs = pd.read_csv(COST_PATH)
    return dict(zip(costs["root"], costs["per_side_cost_bps"], strict=True))


def load_daily_contracts(root: str) -> pd.DataFrame:
    path = RAW_DIR / f"{root}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    frame = (
        pl.scan_parquet(path)
        .filter((pl.col("ts_event") >= START) & (pl.col("ts_event") < END))
        .filter(~pl.col("symbol").str.contains("-"))
        .sort(["symbol", "ts_event"])
        .with_columns(pl.col("ts_event").dt.date().alias("date"))
        .group_by(["date", "symbol"])
        .agg(
            [
                pl.col("close").last().alias("close"),
                pl.col("volume").sum().alias("volume"),
                pl.col("ts_event").last().alias("last_ts"),
            ]
        )
        .collect()
        .to_pandas()
    )
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["last_ts"] = pd.to_datetime(frame["last_ts"], utc=True)
    frame["root"] = root
    frame["months_out"] = [
        contract_months_out(symbol, date_value)
        for symbol, date_value in zip(frame["symbol"], frame["date"], strict=True)
    ]
    frame = frame.dropna(subset=["months_out", "close", "volume"])
    frame = frame[(frame["close"] > 0) & (frame["volume"] >= BASE_MIN_DAILY_VOLUME)]
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def choose_pairs_for_target(
    daily: pd.DataFrame, *, target_months: int, min_volume: float
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in daily.groupby("date", sort=True):
        day = group[group["volume"] >= min_volume].copy()
        early = day[day["months_out"] <= MAX_ANCHOR_MONTHS_OUT]
        if early.empty:
            continue
        anchor = early.sort_values(["volume", "months_out"], ascending=[False, True]).iloc[0]
        far_candidates = day[day["months_out"] - anchor["months_out"] >= target_months]
        if far_candidates.empty:
            continue
        far = far_candidates.sort_values(["months_out", "volume"], ascending=[True, False]).iloc[0]
        months_from_anchor = float(far["months_out"] - anchor["months_out"])
        log_spread = math.log(float(far["close"]) / float(anchor["close"]))
        leg_ts_gap_minutes = abs(
            (pd.Timestamp(far["last_ts"]) - pd.Timestamp(anchor["last_ts"])).total_seconds()
        ) / 60.0
        rows.append(
            {
                "root": str(anchor["root"]),
                "date": date,
                "target_months": target_months,
                "min_volume": min_volume,
                "anchor": str(anchor["symbol"]),
                "far": str(far["symbol"]),
                "anchor_months_out": float(anchor["months_out"]),
                "far_months_out": float(far["months_out"]),
                "months_from_anchor": months_from_anchor,
                "anchor_close": float(anchor["close"]),
                "far_close": float(far["close"]),
                "anchor_volume": float(anchor["volume"]),
                "far_volume": float(far["volume"]),
                "anchor_last_ts": anchor["last_ts"],
                "far_last_ts": far["last_ts"],
                "leg_ts_gap_minutes": leg_ts_gap_minutes,
                "log_spread": log_spread,
                "carry_pct_ann": log_spread / (months_from_anchor / 12.0) * 100.0,
            }
        )
    return pd.DataFrame(rows)


def add_next_spread_returns(daily: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    close_lookup = daily.set_index(["date", "symbol"])["close"]
    last_ts_lookup = daily.set_index(["date", "symbol"])["last_ts"]
    dates = sorted(daily["date"].unique())
    next_date = {date: dates[i + 1] for i, date in enumerate(dates[:-1])}

    records = []
    for row in pairs.itertuples(index=False):
        next_day = next_date.get(row.date)
        anchor_next = np.nan
        far_next = np.nan
        next_leg_ts_gap_minutes = np.nan
        spread_return = np.nan
        if next_day is not None:
            anchor_next = close_lookup.get((next_day, row.anchor), np.nan)
            far_next = close_lookup.get((next_day, row.far), np.nan)
            anchor_next_ts = last_ts_lookup.get((next_day, row.anchor), pd.NaT)
            far_next_ts = last_ts_lookup.get((next_day, row.far), pd.NaT)
            if pd.notna(anchor_next_ts) and pd.notna(far_next_ts):
                next_leg_ts_gap_minutes = abs(
                    (pd.Timestamp(far_next_ts) - pd.Timestamp(anchor_next_ts)).total_seconds()
                ) / 60.0
            if np.isfinite(anchor_next) and np.isfinite(far_next):
                far_ret = math.log(float(far_next) / float(row.far_close))
                anchor_ret = math.log(float(anchor_next) / float(row.anchor_close))
                spread_return = far_ret - anchor_ret
        if (
            row.leg_ts_gap_minutes > MAX_LEG_TS_GAP_MINUTES
            or next_leg_ts_gap_minutes > MAX_LEG_TS_GAP_MINUTES
        ):
            spread_return = np.nan
        item = row._asdict()
        item["next_date"] = next_day
        item["anchor_next_close"] = anchor_next
        item["far_next_close"] = far_next
        item["next_leg_ts_gap_minutes"] = next_leg_ts_gap_minutes
        item["spread_return"] = spread_return
        records.append(item)
    return pd.DataFrame(records)


def build_curve_panel() -> pd.DataFrame:
    frames = []
    for root in ROOTS:
        print(f"Building curve panel for {root}", flush=True)
        daily = load_daily_contracts(root)
        for min_volume in MIN_VOLUME_VARIANTS:
            for target_months in TARGET_MONTHS:
                pairs = choose_pairs_for_target(
                    daily, target_months=target_months, min_volume=min_volume
                )
                if pairs.empty:
                    continue
                frames.append(add_next_spread_returns(daily, pairs))
    if not frames:
        raise RuntimeError("No curve panel rows were generated.")
    panel = pd.concat(frames, ignore_index=True).sort_values(
        ["root", "min_volume", "target_months", "date"]
    )
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    panel["next_date"] = pd.to_datetime(panel["next_date"], utc=True)
    return panel.reset_index(drop=True)


def directional_position(  # noqa: PLR0911
    previous: int, z_score: float, *, entry_z: float, exit_z: float, side_mode: str
) -> int:
    allow_backwardation = side_mode in {"both", "backwardation_only"}
    allow_contango = side_mode in {"both", "contango_only"}
    if not np.isfinite(z_score):
        return 0
    if previous == 0:
        if allow_backwardation and z_score <= -entry_z:
            return 1
        if allow_contango and z_score >= entry_z:
            return -1
        return 0
    if previous > 0:
        if allow_contango and z_score >= entry_z:
            return -1
        if z_score >= -exit_z:
            return 0
        return 1
    if allow_backwardation and z_score <= -entry_z:
        return 1
    if z_score <= exit_z:
        return 0
    return -1


def with_lagged_zscore(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    data = frame.sort_values("date").copy()
    rolling = data["carry_pct_ann"].shift(1).rolling(lookback, min_periods=MIN_Z_OBSERVATIONS)
    data["carry_mean_lagged"] = rolling.mean()
    data["carry_std_lagged"] = rolling.std(ddof=0)
    data["carry_z"] = (data["carry_pct_ann"] - data["carry_mean_lagged"]) / data[
        "carry_std_lagged"
    ]
    data.loc[data["carry_std_lagged"] <= 0, "carry_z"] = np.nan
    return data


def simulate_root(
    frame: pd.DataFrame, variant: Variant, *, root: str, leg_cost_bps: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = with_lagged_zscore(frame, variant.lookback)
    rows = []
    events = []
    previous_position = 0
    previous_anchor = ""
    previous_far = ""
    current_event: dict[str, object] | None = None

    for row in data.itertuples(index=False):
        new_position = directional_position(
            previous_position,
            float(row.carry_z),
            entry_z=variant.entry_z,
            exit_z=variant.exit_z,
            side_mode=variant.side_mode,
        )
        if not np.isfinite(row.spread_return):
            new_position = 0

        if previous_position not in (0, new_position) and current_event:
            current_event["exit_date"] = row.date
            current_event["exit_z"] = row.carry_z
            events.append(current_event)
            current_event = None

        if previous_position == 0 and new_position != 0:
            current_event = {
                "variant": variant.name,
                "root": root,
                "target_months": variant.target_months,
                "min_volume": variant.min_volume,
                "cost_multiplier": variant.cost_multiplier,
                "side": "fade_backwardation" if new_position > 0 else "fade_contango",
                "entry_date": row.date,
                "entry_anchor": row.anchor,
                "entry_far": row.far,
                "entry_carry_pct_ann": row.carry_pct_ann,
                "entry_z": row.carry_z,
                "gross_spread_return": 0.0,
                "weighted_gross_return": 0.0,
                "weighted_cost_return": 0.0,
                "weighted_net_return": 0.0,
                "holding_days": 0,
                "rolls": 0,
            }

        pair_changed = (
            previous_position != 0
            and new_position != 0
            and (row.anchor != previous_anchor or row.far != previous_far)
        )
        spread_turnover = abs(new_position - previous_position)
        leg_cost = (
            2.0
            * leg_cost_bps
            * variant.cost_multiplier
            / 10_000.0
            * spread_turnover
            * ROOT_WEIGHT
        )
        if pair_changed and new_position == previous_position:
            leg_cost += (
                4.0
                * leg_cost_bps
                * variant.cost_multiplier
                / 10_000.0
                * abs(new_position)
                * ROOT_WEIGHT
            )

        gross = new_position * float(row.spread_return) * ROOT_WEIGHT
        net = gross - leg_cost
        if current_event and new_position != 0:
            current_event["gross_spread_return"] = float(
                current_event["gross_spread_return"]
            ) + new_position * float(row.spread_return)
            current_event["weighted_gross_return"] = float(
                current_event["weighted_gross_return"]
            ) + gross
            current_event["weighted_cost_return"] = float(
                current_event["weighted_cost_return"]
            ) + leg_cost
            current_event["weighted_net_return"] = float(current_event["weighted_net_return"]) + net
            current_event["holding_days"] = int(current_event["holding_days"]) + 1
            current_event["rolls"] = int(current_event["rolls"]) + int(pair_changed)

        rows.append(
            {
                "date": row.date,
                "next_date": row.next_date,
                "variant": variant.name,
                "root": root,
                "target_months": variant.target_months,
                "min_volume": variant.min_volume,
                "cost_multiplier": variant.cost_multiplier,
                "anchor": row.anchor,
                "far": row.far,
                "carry_pct_ann": row.carry_pct_ann,
                "carry_z": row.carry_z,
                "spread_return": row.spread_return,
                "position": new_position,
                "pair_changed": pair_changed,
                "gross_return": gross,
                "cost_return": leg_cost,
                "net_return": net,
                "spread_exposure": abs(new_position) * ROOT_WEIGHT,
                "leg_gross_exposure": abs(new_position) * 2.0 * ROOT_WEIGHT,
                "turnover": spread_turnover * ROOT_WEIGHT,
            }
        )
        previous_position = new_position
        previous_anchor = row.anchor
        previous_far = row.far

    if current_event:
        current_event["exit_date"] = data["date"].iloc[-1]
        current_event["exit_z"] = data["carry_z"].iloc[-1]
        current_event["forced_exit"] = True
        events.append(current_event)
    for event in events:
        event.setdefault("forced_exit", False)
        event["duration_days"] = (
            pd.Timestamp(event["exit_date"]) - pd.Timestamp(event["entry_date"])
        ).days
    return pd.DataFrame(rows), pd.DataFrame(events)


def max_drawdown(series: pd.Series) -> float:
    cumulative = series.fillna(0.0).cumsum()
    return float((cumulative - cumulative.cummax()).min())


def periods_per_year(index: pd.Index) -> float:
    dates = pd.DatetimeIndex(index).sort_values()
    if len(dates) < MIN_PERIODS_FOR_ANNUALIZATION:
        return 365.25
    years = max((dates.max() - dates.min()).days / 365.25, 1e-9)
    return len(dates) / years


def summarize_strategy(
    returns: pd.DataFrame, events: pd.DataFrame, variant: Variant
) -> dict[str, object]:
    pp_year = periods_per_year(returns.index)
    net = returns["net_return"].fillna(0.0)
    gross = returns["gross_return"].fillna(0.0)
    costs = returns["cost_return"].fillna(0.0)
    net_std = net.std(ddof=1)
    event_net = events["weighted_net_return"] if not events.empty else pd.Series(dtype=float)
    event_std = event_net.std(ddof=1) if len(event_net) > 1 else np.nan
    return {
        "variant": variant.name,
        "target_months": variant.target_months,
        "min_volume": variant.min_volume,
        "lookback": variant.lookback,
        "entry_z": variant.entry_z,
        "exit_z": variant.exit_z,
        "side_mode": variant.side_mode,
        "cost_multiplier": variant.cost_multiplier,
        "gross_return": gross.sum(),
        "cost_return": costs.sum(),
        "net_return": net.sum(),
        "ann_return": net.mean() * pp_year,
        "ann_vol": net_std * math.sqrt(pp_year) if np.isfinite(net_std) else np.nan,
        "sharpe": net.mean() / net_std * math.sqrt(pp_year) if net_std > 0 else np.nan,
        "tstat": net.mean() / net_std * math.sqrt(len(net)) if net_std > 0 else np.nan,
        "max_drawdown": max_drawdown(net),
        "mean_spread_exposure": returns["spread_exposure"].mean(),
        "mean_leg_gross_exposure": returns["leg_gross_exposure"].mean(),
        "mean_turnover": returns["turnover"].mean(),
        "active_fraction": (returns["spread_exposure"] > 0).mean(),
        "event_count": len(events),
        "event_win_rate": (event_net > 0).mean() if len(event_net) else np.nan,
        "mean_event_net_return": event_net.mean() if len(event_net) else np.nan,
        "event_tstat": event_net.mean() / event_std * math.sqrt(len(event_net))
        if np.isfinite(event_std) and event_std > 0
        else np.nan,
        "bars": len(returns),
        "periods_per_year": pp_year,
    }


def aggregate_variant_returns(root_returns: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(root_returns, ignore_index=True)
    numeric = [
        "gross_return",
        "cost_return",
        "net_return",
        "spread_exposure",
        "leg_gross_exposure",
        "turnover",
    ]
    aggregated = frame.groupby("date", sort=True)[numeric].sum()
    active_roots = frame.assign(active=frame["position"].ne(0)).groupby("date")["active"].sum()
    aggregated["active_roots"] = active_roots
    return aggregated


def run_backtests(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    costs = load_costs()
    metric_rows = []
    all_returns = []
    all_events = []
    panel_groups = {
        key: group.copy()
        for key, group in panel.groupby(["root", "min_volume", "target_months"], sort=False)
    }

    variants = [
        Variant(target, min_volume, lookback, entry_z, exit_z, side_mode, cost_multiplier)
        for target in TARGET_MONTHS
        for min_volume in MIN_VOLUME_VARIANTS
        for lookback in LOOKBACKS
        for entry_z in ENTRY_ZS
        for exit_z in EXIT_ZS
        for side_mode in SIDE_MODES
        for cost_multiplier in COST_MULTIPLIERS
    ]
    for variant in variants:
        root_returns = []
        root_events = []
        for root in ROOTS:
            root_panel = panel_groups.get((root, variant.min_volume, variant.target_months))
            if root_panel is None or root_panel.empty:
                continue
            returns, events = simulate_root(
                root_panel, variant, root=root, leg_cost_bps=float(costs[root])
            )
            root_returns.append(returns)
            if not events.empty:
                root_events.append(events)
        if not root_returns:
            continue
        variant_returns = aggregate_variant_returns(root_returns)
        events = pd.concat(root_events, ignore_index=True) if root_events else pd.DataFrame()
        metrics = summarize_strategy(variant_returns, events, variant)
        metric_rows.append(metrics)
        variant_returns = variant_returns.reset_index()
        variant_returns["variant"] = variant.name
        all_returns.append(variant_returns)
        if not events.empty:
            all_events.append(events)

    metrics = pd.DataFrame(metric_rows).sort_values("net_return", ascending=False)
    returns = pd.concat(all_returns, ignore_index=True)
    events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    return metrics, returns, events


def split_metrics(best_returns: pd.DataFrame, best_events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = best_returns.set_index("date").sort_index()
    splits = {"full": frame}
    for year, group in frame.groupby(frame.index.year):
        splits[str(year)] = group
    for name, data in splits.items():
        if data.empty:
            continue
        net = data["net_return"]
        pp_year = periods_per_year(data.index)
        std = net.std(ddof=1)
        event_slice = best_events
        if name != "full" and not best_events.empty:
            entry_year = pd.to_datetime(best_events["entry_date"], utc=True).dt.year
            event_slice = best_events[entry_year == int(name)]
        rows.append(
            {
                "split": name,
                "gross_return": data["gross_return"].sum(),
                "cost_return": data["cost_return"].sum(),
                "net_return": net.sum(),
                "sharpe": net.mean() / std * math.sqrt(pp_year) if std > 0 else np.nan,
                "tstat": net.mean() / std * math.sqrt(len(net)) if std > 0 else np.nan,
                "max_drawdown": max_drawdown(net),
                "events": len(event_slice),
                "bars": len(data),
            }
        )
    return pd.DataFrame(rows)


def root_event_summary(events: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    selected = events[events["variant"] == variant_name].copy()
    if selected.empty:
        return pd.DataFrame()
    grouped = selected.groupby(["root", "side"], dropna=False)
    rows = []
    for (root, side), group in grouped:
        values = group["weighted_net_return"]
        std = values.std(ddof=1)
        rows.append(
            {
                "root": root,
                "side": side,
                "events": len(group),
                "mean_weighted_net_return": values.mean(),
                "mean_gross_spread_return": group["gross_spread_return"].mean(),
                "win_rate": (values > 0).mean(),
                "event_tstat": values.mean() / std * math.sqrt(len(values)) if std > 0 else np.nan,
                "mean_duration_days": group["duration_days"].mean(),
                "mean_rolls": group["rolls"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["root", "side"])


def robustness_tables(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "target_months",
        "min_volume",
        "cost_multiplier",
        "variant",
        "net_return",
        "cost_return",
        "sharpe",
        "tstat",
        "max_drawdown",
        "event_count",
        "event_tstat",
        "active_fraction",
    ]
    volume_cost_idx = metrics.groupby(["min_volume", "cost_multiplier"])["net_return"].idxmax()
    volume_cost = metrics.loc[volume_cost_idx, columns].sort_values(
        ["min_volume", "cost_multiplier"]
    )

    one_x = metrics[metrics["cost_multiplier"].eq(1.0)]
    target_volume_idx = one_x.groupby(["target_months", "min_volume"])["net_return"].idxmax()
    target_volume = one_x.loc[target_volume_idx, columns].sort_values(
        ["target_months", "min_volume"]
    )
    return volume_cost, target_volume


def plot_best_equity(best_returns: pd.DataFrame, output_path: Path) -> None:
    frame = best_returns.set_index("date").sort_index()
    cumulative_net = frame["net_return"].fillna(0.0).cumsum()
    cumulative_gross = frame["gross_return"].fillna(0.0).cumsum()
    drawdown = cumulative_net - cumulative_net.cummax()

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    axes[0].plot(cumulative_gross.index, cumulative_gross, label="gross spread return", lw=1.2)
    axes[0].plot(cumulative_net.index, cumulative_net, label="net after leg costs", lw=1.5)
    axes[0].axhline(0, color="#333333", lw=0.8)
    axes[0].set_ylabel("Cumulative log return")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.25)

    axes[1].fill_between(drawdown.index, drawdown, 0, color="#c43d3d", alpha=0.25)
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_root_events(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    labels = summary["root"] + "\n" + summary["side"].str.replace("_", "\n")
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    colors = [COLORS.get(root, "#666666") for root in summary["root"]]
    ax.bar(labels, summary["mean_gross_spread_return"] * 10_000, color=colors, alpha=0.85)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_ylabel("Mean gross event spread return, bp")
    ax.set_title("Best variant event returns by metal and side")
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_top_variants(metrics: pd.DataFrame, output_path: Path) -> None:
    top = metrics.head(15).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.barh(top["variant"], top["net_return"], color="#2f7d8c", alpha=0.85)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Net cumulative log return")
    ax.set_title("Top convenience-yield basis variants")
    ax.grid(True, axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(
    *,
    metrics: pd.DataFrame,
    split: pd.DataFrame,
    root_summary: pd.DataFrame,
    volume_cost: pd.DataFrame,
    target_volume: pd.DataFrame,
    best_variant: str,
    output_dir: Path,
) -> None:
    best = metrics.iloc[0]
    top_cols = [
        "variant",
        "min_volume",
        "cost_multiplier",
        "net_return",
        "cost_return",
        "sharpe",
        "tstat",
        "max_drawdown",
        "event_count",
        "event_tstat",
        "active_fraction",
    ]
    lines = [
        "# HYP-0030 Metals Convenience-Yield Basis Backtest",
        "",
        "## Design",
        "",
        "- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.",
        "- Data: raw 1-minute outright futures, collapsed to daily contract closes.",
        "- Pair selection: liquid front contract versus first deferred contract at least the "
        "target tenor away.",
        "- Robustness controls: minimum leg volume variants, cost multipliers, and a "
        f"`{MAX_LEG_TS_GAP_MINUTES:.0f}` minute max near/far last-trade timestamp gap.",
        "- Signal: annualized `log(far/front)` carry z-score using lagged rolling statistics.",
        "- Low carry / backwardation shock: long deferred, short front, expecting carry "
        "normalization.",
        "- High carry / contango shock: short deferred, long front, expecting carry normalization.",
        "- Exit: event-based normalization of the carry z-score; no fixed holding time.",
        "- Costs: two-leg futures execution cost using prior per-side MBP1 estimates.",
        "",
        "## Coverage",
        "",
        f"- Start: `{START.date()}`.",
        f"- End: `{(END - pd.Timedelta(days=1)).date()}`.",
        "",
        "## Best Variant",
        "",
        metrics.head(1)[top_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Volume And Cost Robustness",
        "",
        volume_cost.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 1x Cost Tenor And Volume Robustness",
        "",
        target_volume.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Split Metrics",
        "",
        split.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Event Summary For Best Variant",
        "",
        root_summary.to_markdown(index=False, floatfmt=".4f")
        if not root_summary.empty
        else "No events.",
        "",
        "## Interpretation",
        "",
        (
            f"The best variant is `{best_variant}` with net cumulative log return "
            f"`{best['net_return']:.4f}`, t-stat `{best['tstat']:.2f}`, and "
            f"event t-stat `{best['event_tstat']:.2f}`."
        ),
        "",
        "This is a futures-only test of curve-basis mean reversion. It does not prove a pure "
        "physical arbitrage because spot storage, delivery optionality, warehouse location, and "
        "financing are not directly traded here.",
        "",
        "## Files",
        "",
        "- `curve_panel.parquet`",
        "- `strategy_metrics.csv`",
        "- `best_strategy_returns.csv`",
        "- `event_log.csv`",
        "- `split_metrics.csv`",
        "- `root_event_summary.csv`",
        "- `volume_cost_robustness.csv`",
        "- `target_volume_robustness_1x.csv`",
        "- `best_strategy_equity.png`",
        "- `root_event_summary.png`",
        "- `top_variant_metrics.png`",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUTPUT_DIR / "curve_panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        required_columns = {"root", "min_volume", "target_months", "date", "spread_return"}
        if not required_columns.issubset(panel.columns):
            panel = build_curve_panel()
    else:
        panel = build_curve_panel()
    panel.to_parquet(OUTPUT_DIR / "curve_panel.parquet", index=False)
    panel.head(10_000).to_csv(OUTPUT_DIR / "curve_panel_sample.csv", index=False)

    metrics, returns, events = run_backtests(panel)
    metrics.to_csv(OUTPUT_DIR / "strategy_metrics.csv", index=False)
    returns.to_csv(OUTPUT_DIR / "all_strategy_returns.csv", index=False)
    events.to_csv(OUTPUT_DIR / "event_log.csv", index=False)

    best_variant = str(metrics.iloc[0]["variant"])
    best_returns = returns[returns["variant"] == best_variant].copy()
    best_events = events[events["variant"] == best_variant].copy()
    split = split_metrics(best_returns, best_events)
    root_summary = root_event_summary(events, best_variant)
    volume_cost, target_volume = robustness_tables(metrics)

    best_returns.to_csv(OUTPUT_DIR / "best_strategy_returns.csv", index=False)
    split.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)
    root_summary.to_csv(OUTPUT_DIR / "root_event_summary.csv", index=False)
    volume_cost.to_csv(OUTPUT_DIR / "volume_cost_robustness.csv", index=False)
    target_volume.to_csv(OUTPUT_DIR / "target_volume_robustness_1x.csv", index=False)

    plot_best_equity(best_returns, OUTPUT_DIR / "best_strategy_equity.png")
    plot_root_events(root_summary, OUTPUT_DIR / "root_event_summary.png")
    plot_top_variants(metrics, OUTPUT_DIR / "top_variant_metrics.png")
    write_report(
        metrics=metrics,
        split=split,
        root_summary=root_summary,
        volume_cost=volume_cost,
        target_volume=target_volume,
        best_variant=best_variant,
        output_dir=OUTPUT_DIR,
    )
    print(metrics.head(12).round(4).to_string(index=False))
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
