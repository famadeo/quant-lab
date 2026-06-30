from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS_DIR = Path(
    "/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/continuous"
)
COST_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0015-metals-flow-corrected-residual-reversion"
    / "cost_estimates.csv"
)
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0035-metals-intraday-reversal-backtest"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
TRAIN_FRACTION = 0.70
EMBARGO_DAYS = 5
MIN_TRAILING_OBS = 5
TRAILING_OBS_WINDOW = 30
ENTRY_DELAY_MINUTES = 1
MIN_XS_ASSETS = 3
COST_MULTIPLIERS = [1.0, 2.0, 3.0]
XS_DISPERSION_FILTERS = [0.0, 0.50, 0.70, 0.80]
INDIVIDUAL_TAIL_QS = [0.10, 0.20]
MIN_TSTAT_OBS = 2


@dataclass(frozen=True)
class Rule:
    name: str
    kind: str
    time_utc: str
    signal_window: int
    horizon: int
    root: str = ""

    @property
    def minute_of_day(self) -> int:
        hour, minute = self.time_utc.split(":")
        return int(hour) * 60 + int(minute)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    kind: str
    rule_names: tuple[str, ...]
    cost_multiplier: float
    dispersion_q: float = 0.0
    tail_q: float = 0.20

    @property
    def base_name(self) -> str:
        return self.name.rsplit("_costx", 1)[0]


XS_RULES = [
    Rule("xs_0430_s60_h60", "xs", "04:30", 60, 60),
    Rule("xs_1800_s5_h5", "xs", "18:00", 5, 5),
    Rule("xs_1800_s30_h60", "xs", "18:00", 30, 60),
    Rule("xs_2000_s5_h5", "xs", "20:00", 5, 5),
    Rule("xs_2000_s15_h5", "xs", "20:00", 15, 5),
    Rule("xs_2000_s30_h30", "xs", "20:00", 30, 30),
    Rule("xs_2000_s60_h30", "xs", "20:00", 60, 30),
    Rule("xs_2100_s15_h5", "xs", "21:00", 15, 5),
    Rule("xs_2100_s15_h60", "xs", "21:00", 15, 60),
    Rule("xs_2100_s15_h120", "xs", "21:00", 15, 120),
    Rule("xs_2230_s30_h5", "xs", "22:30", 30, 5),
    Rule("xs_2230_s60_h5", "xs", "22:30", 60, 5),
    Rule("xs_2230_s60_h30", "xs", "22:30", 60, 30),
]

INDIVIDUAL_RULES = [
    Rule("gc_0500_s5_h5", "individual", "05:00", 5, 5, "GC"),
    Rule("gc_1100_s15_h5", "individual", "11:00", 15, 5, "GC"),
    Rule("gc_2100_s15_h15", "individual", "21:00", 15, 15, "GC"),
    Rule("gc_2230_s30_h15", "individual", "22:30", 30, 15, "GC"),
    Rule("si_0430_s60_h60", "individual", "04:30", 60, 60, "SI"),
    Rule("si_2030_s5_h15", "individual", "20:30", 5, 15, "SI"),
    Rule("si_2100_s15_h15", "individual", "21:00", 15, 15, "SI"),
    Rule("si_2300_s5_h5", "individual", "23:00", 5, 5, "SI"),
    Rule("hg_0430_s60_h60", "individual", "04:30", 60, 60, "HG"),
    Rule("hg_2100_s15_h120", "individual", "21:00", 15, 120, "HG"),
    Rule("hg_2100_s15_h60", "individual", "21:00", 15, 60, "HG"),
    Rule("hg_2100_s30_h120", "individual", "21:00", 30, 120, "HG"),
    Rule("pl_0300_s5_h5", "individual", "03:00", 5, 5, "PL"),
    Rule("pl_1300_s30_h5", "individual", "13:00", 30, 5, "PL"),
    Rule("pl_1300_s60_h5", "individual", "13:00", 60, 5, "PL"),
    Rule("pl_2230_s60_h5", "individual", "22:30", 60, 5, "PL"),
    Rule("pl_2300_s5_h5", "individual", "23:00", 5, 5, "PL"),
    Rule("pa_0900_s5_h5", "individual", "09:00", 5, 5, "PA"),
    Rule("pa_2000_s5_h5", "individual", "20:00", 5, 5, "PA"),
    Rule("pa_2000_s5_h15", "individual", "20:00", 5, 15, "PA"),
    Rule("pa_2100_s5_h5", "individual", "21:00", 5, 5, "PA"),
]

XS_BUNDLES = {
    "xs_core": (
        "xs_0430_s60_h60",
        "xs_1800_s30_h60",
        "xs_2000_s5_h5",
        "xs_2100_s15_h120",
        "xs_2230_s60_h5",
    ),
    "xs_fast": (
        "xs_1800_s5_h5",
        "xs_2000_s5_h5",
        "xs_2100_s15_h5",
        "xs_2230_s60_h5",
    ),
    "xs_long": (
        "xs_0430_s60_h60",
        "xs_1800_s30_h60",
        "xs_2000_s60_h30",
        "xs_2100_s15_h120",
        "xs_2230_s60_h30",
    ),
}

INDIVIDUAL_BUNDLES = {
    root.lower() + "_core": tuple(rule.name for rule in INDIVIDUAL_RULES if rule.root == root)
    for root in ROOTS
}


def safe_tstat(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < MIN_TSTAT_OBS:
        return np.nan
    std = float(clean.std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        return np.nan
    return float(clean.mean() / std * math.sqrt(len(clean)))


def max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return np.nan
    cumulative = values.fillna(0.0).cumsum()
    drawdown = cumulative - cumulative.cummax()
    return float(drawdown.min())


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < MIN_TSTAT_OBS:
        return np.nan
    elapsed_years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
    if elapsed_years <= 0:
        return np.nan
    return float(len(index) / elapsed_years)


def fmt_cost(value: float) -> str:
    return str(value).replace(".", "p")


def fmt_q(value: float) -> str:
    return str(value).replace(".", "p")


def load_costs() -> pd.Series:
    if COST_PATH.exists():
        costs = pd.read_csv(COST_PATH).set_index("root")["per_side_cost_bps"]
    else:
        costs = pd.Series(
            {"GC": 0.55078817788255, "SI": 1.8695083193124464, "HG": 0.8003841844080726,
             "PL": 2.563182447326601, "PA": 5.593884020137983},
            name="per_side_cost_bps",
        )
    costs = costs.reindex(ROOTS).astype(float)
    if costs.isna().any():
        missing = costs[costs.isna()].index.tolist()
        raise ValueError(f"missing cost estimates for {missing}")
    return costs


def load_minute_data(root: str) -> tuple[pd.DataFrame, dict[str, object]]:
    path = CONTINUOUS_DIR / f"{root}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = (
        pl.scan_parquet(path)
        .select("ts", "cont_logprice", "cont_close", "volume", "is_roll")
        .collect()
        .to_pandas()
    )
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw = raw.sort_values("ts").replace([np.inf, -np.inf], np.nan)
    roll_dates = set(raw.loc[raw["is_roll"].fillna(False), "ts"].dt.normalize())
    full_index = pd.date_range(raw["ts"].min(), raw["ts"].max(), freq="1min", tz="UTC")
    minute = raw.set_index("ts")[["cont_logprice", "volume"]].reindex(full_index)
    observed = minute["cont_logprice"].notna()
    minute["logp"] = minute["cont_logprice"].ffill()
    minute["observed"] = observed
    minute["volume"] = minute["volume"].fillna(0.0)
    minute = minute.drop(columns=["cont_logprice"])
    minute["date"] = minute.index.normalize()
    if roll_dates:
        minute = minute.loc[~minute["date"].isin(roll_dates)].copy()
    minute = minute.dropna(subset=["logp"])
    minute["obs_30m"] = minute.groupby("date", sort=False)["observed"].transform(
        lambda values: values.rolling(TRAILING_OBS_WINDOW, min_periods=1).sum()
    )
    inventory = {
        "root": root,
        "rows": len(raw),
        "first_ts": raw["ts"].min(),
        "last_ts": raw["ts"].max(),
        "roll_dates": len(roll_dates),
        "reindexed_rows": len(minute),
        "observed_fraction": float(minute["observed"].mean()),
        "median_obs_30m": float(minute["obs_30m"].median()),
    }
    return minute[["logp", "observed", "obs_30m", "volume", "date"]], inventory


def decision_timestamps(dates: pd.DatetimeIndex, minute_of_day: int) -> pd.DatetimeIndex:
    return dates + pd.to_timedelta(minute_of_day, unit="m")


def rule_arrays(
    minute_data: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    rule: Rule,
) -> dict[str, pd.DataFrame]:
    decision_ts = decision_timestamps(dates, rule.minute_of_day)
    prior_ts = decision_ts - pd.to_timedelta(rule.signal_window, unit="m")
    entry_ts = decision_ts + pd.to_timedelta(ENTRY_DELAY_MINUTES, unit="m")
    exit_ts = entry_ts + pd.to_timedelta(rule.horizon, unit="m")
    same_day = exit_ts.normalize() == decision_ts.normalize()
    out: dict[str, pd.DataFrame] = {}
    for root, minute in minute_data.items():
        decision = minute.reindex(decision_ts)
        prior = minute.reindex(prior_ts)
        entry = minute.reindex(entry_ts)
        exit_ = minute.reindex(exit_ts)
        valid = (
            same_day
            & decision["observed"].to_numpy(dtype=bool, na_value=False)
            & prior["observed"].to_numpy(dtype=bool, na_value=False)
            & entry["observed"].to_numpy(dtype=bool, na_value=False)
            & exit_["observed"].to_numpy(dtype=bool, na_value=False)
            & (decision["obs_30m"].fillna(0.0).to_numpy(dtype=float) >= MIN_TRAILING_OBS)
        )
        signal = decision["logp"].to_numpy(dtype=float) - prior["logp"].to_numpy(dtype=float)
        forward = exit_["logp"].to_numpy(dtype=float) - entry["logp"].to_numpy(dtype=float)
        frame = pd.DataFrame(
            {
                "date": dates,
                "decision_ts": decision_ts,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "signal": signal,
                "forward_return": forward,
                "valid": valid
                & np.isfinite(signal)
                & np.isfinite(forward),
            }
        )
        out[root] = frame
    return out


def generate_xs_rule_events(
    rule: Rule,
    arrays: dict[str, pd.DataFrame],
    costs_bps: pd.Series,
) -> pd.DataFrame:
    base = next(iter(arrays.values()))[["date", "decision_ts", "entry_ts", "exit_ts"]].copy()
    signal = pd.DataFrame({root: arrays[root]["signal"].to_numpy(float) for root in ROOTS})
    fwd = pd.DataFrame({root: arrays[root]["forward_return"].to_numpy(float) for root in ROOTS})
    valid = pd.DataFrame({root: arrays[root]["valid"].to_numpy(bool) for root in ROOTS})
    signal = signal.where(valid)
    fwd = fwd.where(valid)
    rows: list[dict[str, object]] = []
    for idx, row in base.iterrows():
        sig = signal.iloc[idx].dropna()
        ret = fwd.iloc[idx].dropna()
        roots = sorted(set(sig.index).intersection(ret.index))
        if len(roots) < MIN_XS_ASSETS:
            continue
        sig = sig.loc[roots]
        ret = ret.loc[roots]
        winner = str(sig.idxmax())
        loser = str(sig.idxmin())
        if winner == loser:
            continue
        weights = {root: 0.0 for root in ROOTS}
        weights[loser] = 0.5
        weights[winner] = -0.5
        gross = 0.5 * float(ret[loser]) - 0.5 * float(ret[winner])
        cost_1x = 2.0 * (
            0.5 * float(costs_bps[loser]) + 0.5 * float(costs_bps[winner])
        ) / 10_000.0
        item = {
            "rule": rule.name,
            "kind": "xs",
            "root": "",
            "time_utc": rule.time_utc,
            "signal_window": rule.signal_window,
            "horizon": rule.horizon,
            "date": row["date"],
            "decision_ts": row["decision_ts"],
            "entry_ts": row["entry_ts"],
            "exit_ts": row["exit_ts"],
            "winner": winner,
            "loser": loser,
            "n_assets": len(roots),
            "signal_dispersion": float(sig[winner] - sig[loser]),
            "trade_signal": float(sig[loser] - sig[winner]),
            "gross_return": gross,
            "cost_1x": cost_1x,
            "gross_exposure": 1.0,
            "turnover": 2.0,
            "holding_minutes": rule.horizon,
        }
        for root_name in ROOTS:
            item[f"weight_{root_name}"] = weights[root_name]
            item[f"signal_{root_name}"] = float(sig[root_name]) if root_name in sig else np.nan
            item[f"fwd_{root_name}"] = float(ret[root_name]) if root_name in ret else np.nan
        rows.append(item)
    return pd.DataFrame(rows)


def generate_individual_rule_events(
    rule: Rule,
    arrays: dict[str, pd.DataFrame],
    costs_bps: pd.Series,
) -> pd.DataFrame:
    frame = arrays[rule.root].copy()
    frame = frame[frame["valid"]].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["rule"] = rule.name
    frame["kind"] = "individual"
    frame["root"] = rule.root
    frame["time_utc"] = rule.time_utc
    frame["signal_window"] = rule.signal_window
    frame["horizon"] = rule.horizon
    frame["base_forward_return"] = frame["forward_return"].astype(float)
    frame["cost_1x"] = 2.0 * float(costs_bps[rule.root]) / 10_000.0
    frame["gross_exposure"] = 1.0
    frame["turnover"] = 2.0
    frame["holding_minutes"] = rule.horizon
    return frame[
        [
            "rule",
            "kind",
            "root",
            "time_utc",
            "signal_window",
            "horizon",
            "date",
            "decision_ts",
            "entry_ts",
            "exit_ts",
            "signal",
            "base_forward_return",
            "cost_1x",
            "gross_exposure",
            "turnover",
            "holding_minutes",
        ]
    ]


def apply_non_overlap(events: pd.DataFrame, rule_priority: dict[str, int]) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    data = events.copy()
    data["priority"] = data["rule"].map(rule_priority).fillna(999).astype(int)
    data = data.sort_values(["entry_ts", "priority", "rule"]).reset_index(drop=True)
    keep = []
    active_until = pd.Timestamp.min.tz_localize("UTC")
    for row in data.itertuples(index=False):
        entry_ts = pd.Timestamp(row.entry_ts)
        exit_ts = pd.Timestamp(row.exit_ts)
        if entry_ts > active_until:
            keep.append(True)
            active_until = exit_ts
        else:
            keep.append(False)
    out = data.loc[keep].drop(columns=["priority"]).copy()
    return out.reset_index(drop=True)


def train_test_dates(dates: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = pd.DatetimeIndex(sorted(pd.Series(dates).dropna().unique()))
    split_idx = min(max(int(len(unique_dates) * TRAIN_FRACTION), 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_idx])
    test_start = split_date + pd.Timedelta(days=EMBARGO_DAYS)
    return split_date, test_start


def build_xs_variant_events(
    spec: VariantSpec,
    raw_xs_events: pd.DataFrame,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    events = raw_xs_events[raw_xs_events["rule"].isin(spec.rule_names)].copy()
    if events.empty:
        return events
    if spec.dispersion_q > 0:
        train = events[events["date"] < split_date]
        thresholds = train.groupby("rule")["signal_dispersion"].quantile(spec.dispersion_q)
        events["dispersion_threshold"] = events["rule"].map(thresholds)
        events = events[
            events["dispersion_threshold"].notna()
            & (events["signal_dispersion"] >= events["dispersion_threshold"])
        ].copy()
    else:
        events["dispersion_threshold"] = np.nan
    priorities = {rule: idx for idx, rule in enumerate(spec.rule_names)}
    events = apply_non_overlap(events, priorities)
    events["variant"] = spec.name
    events["cost_multiplier"] = spec.cost_multiplier
    events["cost_return"] = events["cost_1x"] * spec.cost_multiplier
    events["net_return"] = events["gross_return"] - events["cost_return"]
    return events


def build_individual_variant_events(
    spec: VariantSpec,
    raw_individual_events: pd.DataFrame,
    split_date: pd.Timestamp,
) -> pd.DataFrame:
    events = raw_individual_events[raw_individual_events["rule"].isin(spec.rule_names)].copy()
    if events.empty:
        return events
    train = events[events["date"] < split_date]
    low = train.groupby("rule")["signal"].quantile(spec.tail_q)
    high = train.groupby("rule")["signal"].quantile(1.0 - spec.tail_q)
    events["low_threshold"] = events["rule"].map(low)
    events["high_threshold"] = events["rule"].map(high)
    events["direction"] = 0
    events.loc[events["signal"] <= events["low_threshold"], "direction"] = 1
    events.loc[events["signal"] >= events["high_threshold"], "direction"] = -1
    events = events[events["direction"].ne(0)].copy()
    events["gross_return"] = events["direction"] * events["base_forward_return"]
    for root in ROOTS:
        events[f"weight_{root}"] = 0.0
    for idx, row in events.iterrows():
        events.at[idx, f"weight_{row['root']}"] = float(row["direction"])
    priorities = {rule: idx for idx, rule in enumerate(spec.rule_names)}
    events = apply_non_overlap(events, priorities)
    events["variant"] = spec.name
    events["cost_multiplier"] = spec.cost_multiplier
    events["cost_return"] = events["cost_1x"] * spec.cost_multiplier
    events["net_return"] = events["gross_return"] - events["cost_return"]
    return events


def daily_returns(events: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=all_dates)
    out.index.name = "date"
    if events.empty:
        out["gross_return"] = 0.0
        out["cost_return"] = 0.0
        out["net_return"] = 0.0
        out["gross_exposure"] = 0.0
        out["turnover"] = 0.0
        out["event_count"] = 0.0
        return out
    grouped = events.groupby("date").agg(
        gross_return=("gross_return", "sum"),
        cost_return=("cost_return", "sum"),
        net_return=("net_return", "sum"),
        gross_exposure=("gross_exposure", "sum"),
        turnover=("turnover", "sum"),
        event_count=("rule", "size"),
    )
    out = out.join(grouped, how="left").fillna(0.0)
    return out


def summarize_daily(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    *,
    label: str,
    periods_per_year: float,
) -> dict[str, float | str]:
    if daily.empty:
        return {
            "split": label,
            "gross_return": np.nan,
            "cost_return": np.nan,
            "net_return": np.nan,
            "cagr": np.nan,
            "gross_to_cost": np.nan,
            "tstat": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "event_count": 0,
            "event_win_rate": np.nan,
            "mean_event_net": np.nan,
            "mean_events_per_day": np.nan,
            "active_day_fraction": np.nan,
            "turnover": np.nan,
        }
    net = daily["net_return"].astype(float)
    gross = daily["gross_return"].astype(float)
    cost = daily["cost_return"].astype(float)
    std = float(net.std(ddof=1)) if len(net) > 1 else np.nan
    years = (
        (pd.Timestamp(daily.index[-1]) - pd.Timestamp(daily.index[0])).total_seconds()
        / (365.25 * 24 * 60 * 60)
        if len(daily) > 1
        else np.nan
    )
    net_sum = float(net.sum())
    cagr = math.exp(net_sum / years) - 1.0 if years and years > 0 else np.nan
    event_net = events["net_return"].astype(float) if not events.empty else pd.Series(dtype=float)
    return {
        "split": label,
        "gross_return": float(gross.sum()),
        "cost_return": float(cost.sum()),
        "net_return": net_sum,
        "cagr": cagr,
        "gross_to_cost": float(gross.sum() / cost.sum()) if cost.sum() > 0 else np.nan,
        "tstat": safe_tstat(net),
        "sharpe": float(net.mean() / std * math.sqrt(periods_per_year))
        if np.isfinite(std) and std > 0
        else np.nan,
        "max_drawdown": max_drawdown(net),
        "event_count": len(events),
        "event_win_rate": float((event_net > 0).mean()) if len(event_net) else np.nan,
        "mean_event_net": float(event_net.mean()) if len(event_net) else np.nan,
        "mean_events_per_day": float(daily["event_count"].mean()),
        "active_day_fraction": float((daily["event_count"] > 0).mean()),
        "turnover": float(daily["turnover"].sum()),
    }


def split_frames(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    split_date: pd.Timestamp,
    test_start: pd.Timestamp,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    return {
        "train": (daily[daily.index < split_date], events[events["date"] < split_date]),
        "embargo": (
            daily[(daily.index >= split_date) & (daily.index < test_start)],
            events[(events["date"] >= split_date) & (events["date"] < test_start)],
        ),
        "test": (daily[daily.index >= test_start], events[events["date"] >= test_start]),
        "full": (daily, events),
    }


def score_train(metrics: dict[str, float | str]) -> float:
    net = float(metrics.get("net_return", np.nan))
    gross_to_cost = float(metrics.get("gross_to_cost", np.nan))
    tstat = float(metrics.get("tstat", np.nan))
    if not all(np.isfinite([net, gross_to_cost, tstat])):
        return -np.inf
    if net <= 0.0 or gross_to_cost <= 1.0:
        return -np.inf
    return float(tstat + 0.25 * math.log1p(max(gross_to_cost, 0.0)) + 10.0 * net)


def evaluate_variant(
    spec: VariantSpec,
    raw_xs_events: pd.DataFrame,
    raw_individual_events: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    split_date: pd.Timestamp,
    test_start: pd.Timestamp,
    periods_per_year: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if spec.kind == "xs":
        events = build_xs_variant_events(spec, raw_xs_events, split_date)
    else:
        events = build_individual_variant_events(spec, raw_individual_events, split_date)
    daily = daily_returns(events, all_dates)
    split_data = split_frames(daily, events, split_date, test_start)
    split_rows = {}
    for split, (daily_slice, event_slice) in split_data.items():
        split_rows[split] = summarize_daily(
            daily_slice,
            event_slice,
            label=split,
            periods_per_year=periods_per_year,
        )
    train_score = score_train(split_rows["train"])
    row: dict[str, object] = {
        "variant": spec.name,
        "base_variant": spec.base_name,
        "kind": spec.kind,
        "rule_count": len(spec.rule_names),
        "rules": ",".join(spec.rule_names),
        "cost_multiplier": spec.cost_multiplier,
        "dispersion_q": spec.dispersion_q,
        "tail_q": spec.tail_q,
        "train_score": train_score,
    }
    for split, stats in split_rows.items():
        for key, value in stats.items():
            if key != "split":
                row[f"{split}_{key}"] = value
    split_metrics = pd.DataFrame(split_rows.values())
    split_metrics.insert(0, "variant", spec.name)
    daily = daily.copy()
    daily["variant"] = spec.name
    return row, daily, events, split_metrics


def build_variants() -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    xs_rule_names = [rule.name for rule in XS_RULES]
    for rule_name in xs_rule_names:
        variants.extend(
            VariantSpec(
                name=(
                    f"{rule_name}_dispq{fmt_q(dispersion_q)}_"
                    f"costx{fmt_cost(cost_multiplier)}"
                ),
                kind="xs",
                rule_names=(rule_name,),
                cost_multiplier=cost_multiplier,
                dispersion_q=dispersion_q,
            )
            for dispersion_q in XS_DISPERSION_FILTERS
            for cost_multiplier in COST_MULTIPLIERS
        )
    for bundle_name, rule_names in XS_BUNDLES.items():
        variants.extend(
            VariantSpec(
                name=(
                    f"{bundle_name}_dispq{fmt_q(dispersion_q)}_"
                    f"costx{fmt_cost(cost_multiplier)}"
                ),
                kind="xs",
                rule_names=rule_names,
                cost_multiplier=cost_multiplier,
                dispersion_q=dispersion_q,
            )
            for dispersion_q in XS_DISPERSION_FILTERS
            for cost_multiplier in COST_MULTIPLIERS
        )
    for rule in INDIVIDUAL_RULES:
        variants.extend(
            VariantSpec(
                name=f"{rule.name}_tailq{fmt_q(tail_q)}_costx{fmt_cost(cost_multiplier)}",
                kind="individual",
                rule_names=(rule.name,),
                cost_multiplier=cost_multiplier,
                tail_q=tail_q,
            )
            for tail_q in INDIVIDUAL_TAIL_QS
            for cost_multiplier in COST_MULTIPLIERS
        )
    for bundle_name, rule_names in INDIVIDUAL_BUNDLES.items():
        variants.extend(
            VariantSpec(
                name=f"{bundle_name}_tailq{fmt_q(tail_q)}_costx{fmt_cost(cost_multiplier)}",
                kind="individual",
                rule_names=rule_names,
                cost_multiplier=cost_multiplier,
                tail_q=tail_q,
            )
            for tail_q in INDIVIDUAL_TAIL_QS
            for cost_multiplier in COST_MULTIPLIERS
        )
    return variants


def generate_raw_events(
    minute_data: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    costs_bps: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xs_events: list[pd.DataFrame] = []
    individual_events: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    all_rules = [*XS_RULES, *INDIVIDUAL_RULES]
    for rule in all_rules:
        arrays = rule_arrays(minute_data, dates, rule)
        if rule.kind == "xs":
            events = generate_xs_rule_events(rule, arrays, costs_bps)
            xs_events.append(events)
            diagnostics.append(
                {
                    "rule": rule.name,
                    "kind": rule.kind,
                    "root": "",
                    "time_utc": rule.time_utc,
                    "signal_window": rule.signal_window,
                    "horizon": rule.horizon,
                    "events": len(events),
                    "first_date": events["date"].min() if not events.empty else pd.NaT,
                    "last_date": events["date"].max() if not events.empty else pd.NaT,
                    "mean_gross_return": events["gross_return"].mean()
                    if not events.empty
                    else np.nan,
                    "gross_tstat": safe_tstat(events["gross_return"])
                    if not events.empty
                    else np.nan,
                }
            )
        else:
            events = generate_individual_rule_events(rule, arrays, costs_bps)
            individual_events.append(events)
            diagnostics.append(
                {
                    "rule": rule.name,
                    "kind": rule.kind,
                    "root": rule.root,
                    "time_utc": rule.time_utc,
                    "signal_window": rule.signal_window,
                    "horizon": rule.horizon,
                    "events": len(events),
                    "first_date": events["date"].min() if not events.empty else pd.NaT,
                    "last_date": events["date"].max() if not events.empty else pd.NaT,
                    "mean_forward_return": events["base_forward_return"].mean()
                    if not events.empty
                    else np.nan,
                    "signal_tstat": safe_tstat(events["signal"]) if not events.empty else np.nan,
                }
            )
    raw_xs = pd.concat(xs_events, ignore_index=True) if xs_events else pd.DataFrame()
    raw_individual = (
        pd.concat(individual_events, ignore_index=True) if individual_events else pd.DataFrame()
    )
    return raw_xs, raw_individual, pd.DataFrame(diagnostics)


def select_variant(metrics: pd.DataFrame) -> pd.Series:
    one_x = metrics[metrics["cost_multiplier"].eq(1.0)].copy()
    viable = one_x[np.isfinite(one_x["train_score"])].copy()
    if viable.empty:
        viable = one_x.copy()
    viable = viable.sort_values(["train_score", "train_net_return"], ascending=False)
    return viable.iloc[0]


def make_cost_sensitivity(metrics: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    base_name = selected["base_variant"]
    return (
        metrics[metrics["base_variant"].eq(base_name)]
        .sort_values("cost_multiplier")
        .reset_index(drop=True)
    )


def robust_1x_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics[
            metrics["cost_multiplier"].eq(1.0)
            & metrics["train_net_return"].gt(0.0)
            & metrics["test_net_return"].gt(0.0)
            & metrics["train_gross_to_cost"].gt(1.5)
            & metrics["test_gross_to_cost"].gt(1.5)
        ]
        .sort_values(["test_tstat", "train_tstat"], ascending=False)
        .reset_index(drop=True)
    )


def cost_2x_survivors(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics[
            metrics["cost_multiplier"].eq(2.0)
            & metrics["train_net_return"].gt(0.0)
            & metrics["test_net_return"].gt(0.0)
        ]
        .sort_values(["test_tstat", "train_tstat"], ascending=False)
        .reset_index(drop=True)
    )


def plot_equity(
    selected_daily: pd.DataFrame, split_date: pd.Timestamp, test_start: pd.Timestamp
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    selected_daily["net_return"].cumsum().plot(ax=ax, lw=1.5, label="net")
    selected_daily["gross_return"].cumsum().plot(ax=ax, lw=1.0, alpha=0.65, label="gross")
    ax.axvline(split_date, color="black", ls="--", lw=1.0, alpha=0.8, label="train split")
    ax.axvline(test_start, color="tab:red", ls=":", lw=1.0, alpha=0.8, label="test start")
    ax.set_title("Selected Intraday Reversal Strategy")
    ax.set_xlabel("")
    ax.set_ylabel("cumulative log return")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(OUTPUT_DIR / "selected_equity.png", dpi=160)
    plt.close(fig)


def plot_train_test(metrics: pd.DataFrame) -> None:
    one_x = metrics[metrics["cost_multiplier"].eq(1.0)].copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    colors = one_x["kind"].map({"xs": "#386cb0", "individual": "#bf5b17"}).fillna("#666666")
    ax.scatter(
        one_x["train_tstat"],
        one_x["test_tstat"],
        s=22,
        c=colors,
        alpha=0.75,
        edgecolors="none",
    )
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("1x Cost Variants: Train vs Test Daily t-stat")
    ax.set_xlabel("train t-stat")
    ax.set_ylabel("test t-stat")
    ax.grid(True, alpha=0.25)
    fig.savefig(OUTPUT_DIR / "train_test_tstat_scatter.png", dpi=160)
    plt.close(fig)


def plot_rule_contribution(selected_events: pd.DataFrame) -> None:
    if selected_events.empty:
        return
    by_rule = (
        selected_events.groupby("rule", as_index=False)
        .agg(net_return=("net_return", "sum"), events=("rule", "size"))
        .sort_values("net_return")
    )
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.barh(by_rule["rule"], by_rule["net_return"], color="#386cb0")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Selected Variant Net Return by Rule")
    ax.set_xlabel("net log return")
    fig.savefig(OUTPUT_DIR / "selected_rule_contribution.png", dpi=160)
    plt.close(fig)


def write_report(
    *,
    inventory: pd.DataFrame,
    raw_diagnostics: pd.DataFrame,
    metrics: pd.DataFrame,
    selected: pd.Series,
    selected_split: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    robust_candidates: pd.DataFrame,
    cost2_survivors: pd.DataFrame,
    split_date: pd.Timestamp,
    test_start: pd.Timestamp,
) -> None:
    metric_cols = [
        "variant",
        "kind",
        "rule_count",
        "cost_multiplier",
        "dispersion_q",
        "tail_q",
        "train_net_return",
        "train_gross_to_cost",
        "train_tstat",
        "train_event_count",
        "test_net_return",
        "test_gross_to_cost",
        "test_tstat",
        "test_event_count",
        "full_net_return",
        "full_cagr",
        "full_sharpe",
        "full_max_drawdown",
    ]
    diag_cols = [
        "rule",
        "kind",
        "root",
        "time_utc",
        "signal_window",
        "horizon",
        "events",
        "mean_gross_return",
        "gross_tstat",
        "mean_forward_return",
        "signal_tstat",
    ]
    top_train = (
        metrics[metrics["cost_multiplier"].eq(1.0)]
        .sort_values(["train_score", "train_net_return"], ascending=False)
        .head(20)
    )
    top_test = (
        metrics[metrics["cost_multiplier"].eq(1.0)]
        .sort_values(["test_tstat", "test_net_return"], ascending=False)
        .head(20)
    )
    lines = [
        "# HYP-0035 Metals Intraday Reversal Backtest",
        "",
        "## Design",
        "",
        "- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.",
        "- Data: 1-minute continuous futures bars from 2016-06-22 through 2026-06-21.",
        "- Signal: prior intraday log return at fixed UTC decision marks.",
        f"- Execution: enter exactly {ENTRY_DELAY_MINUTES} minute after the decision mark and "
        "exit at the specified fixed horizon.",
        "- Tradability filter: decision, prior, entry, and exit bars must all be observed; "
        f"the decision mark must have at least {MIN_TRAILING_OBS} observed bars in the prior "
        f"{TRAILING_OBS_WINDOW} minutes; roll dates are excluded.",
        "- Cross-sectional expression: long the weakest prior-return metal and short the "
        "strongest prior-return metal, dollar-neutral at 100% gross exposure.",
        "- Individual expression: fade root-level prior-return tails using train-sample "
        "10/90 or 20/80 thresholds.",
        "- Non-overlap: combined variants skip new entries while a prior trade is open.",
        "- Costs: MBP1 per-side cost estimates charged on entry and exit turnover; "
        "1x/2x/3x cost sensitivity is reported.",
        f"- Split: train dates before {split_date.date()}, embargo until {test_start.date()}, "
        "test thereafter.",
        "",
        "## Data Inventory",
        "",
        inventory.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Selected Variant",
        "",
        pd.DataFrame([selected[metric_cols].to_dict()]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Selected Split Metrics",
        "",
        selected_split.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Selected Cost Sensitivity",
        "",
        cost_sensitivity[metric_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Robust 1x Candidate Screen",
        "",
        "Positive train and test net return, with train and test gross/cost both above 1.5.",
        "",
        robust_candidates[metric_cols].to_markdown(index=False, floatfmt=".6f")
        if not robust_candidates.empty
        else "No 1x variant passed the robust candidate screen.",
        "",
        "## 2x Cost Survivors",
        "",
        "Positive train and test net return at double the base cost estimate.",
        "",
        cost2_survivors[metric_cols].to_markdown(index=False, floatfmt=".6f")
        if not cost2_survivors.empty
        else "No 2x cost variant had positive train and test net return.",
        "",
        "## Top Train-Selected 1x Variants",
        "",
        top_train[metric_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Top Test 1x Variants",
        "",
        top_test[metric_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw Rule Diagnostics",
        "",
        raw_diagnostics[diag_cols].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "This is a tradability-aware follow-up to HYP-0034. The whitelist itself was motivated "
        "by the prior discovery pass, so the test split is useful but not a pristine untouched "
        "sample. Promotion should require robustness of the selected rule family, not only a "
        "single winning row.",
        "",
        "## Files",
        "",
        "- `variant_metrics.csv`",
        "- `selected_split_metrics.csv`",
        "- `selected_cost_sensitivity.csv`",
        "- `robust_1x_candidates.csv`",
        "- `cost_2x_survivors.csv`",
        "- `selected_daily_returns.csv`",
        "- `selected_events.csv`",
        "- `raw_xs_events.parquet`",
        "- `raw_individual_events.parquet`",
        "- `raw_rule_diagnostics.csv`",
        "- `selected_equity.png`",
        "- `train_test_tstat_scatter.png`",
        "- `selected_rule_contribution.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    costs_bps = load_costs()
    costs_bps.rename("per_side_cost_bps").to_csv(OUTPUT_DIR / "costs_bps.csv")

    minute_data: dict[str, pd.DataFrame] = {}
    inventory_rows: list[dict[str, object]] = []
    for root in ROOTS:
        print(f"Loading {root} 1-minute data", flush=True)
        minute, inventory = load_minute_data(root)
        minute_data[root] = minute
        inventory_rows.append(inventory)
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(OUTPUT_DIR / "data_inventory.csv", index=False)

    all_dates = pd.DatetimeIndex(
        sorted(set().union(*(set(frame["date"].unique()) for frame in minute_data.values())))
    )
    all_dates = pd.DatetimeIndex(all_dates).tz_convert("UTC")
    split_date, test_start = train_test_dates(all_dates)
    periods_per_year = infer_periods_per_year(all_dates)

    print("Generating raw candidate events", flush=True)
    raw_xs_events, raw_individual_events, raw_diagnostics = generate_raw_events(
        minute_data, all_dates, costs_bps
    )
    raw_xs_events.to_parquet(OUTPUT_DIR / "raw_xs_events.parquet", index=False)
    raw_individual_events.to_parquet(OUTPUT_DIR / "raw_individual_events.parquet", index=False)
    raw_diagnostics.to_csv(OUTPUT_DIR / "raw_rule_diagnostics.csv", index=False)

    variants = build_variants()
    print(f"Evaluating {len(variants)} variants", flush=True)
    metric_rows: list[dict[str, object]] = []
    selected_row: pd.Series | None = None

    for spec in variants:
        row, _daily, _events, _split_metrics = evaluate_variant(
            spec,
            raw_xs_events,
            raw_individual_events,
            all_dates,
            split_date,
            test_start,
            periods_per_year,
        )
        metric_rows.append(row)
        row_series = pd.Series(row)
        if spec.cost_multiplier == 1.0 and (
            selected_row is None
            or float(row_series["train_score"]) > float(selected_row["train_score"])
        ):
            selected_row = row_series

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["train_score", "train_net_return"], ascending=False
    )
    metrics.to_csv(OUTPUT_DIR / "variant_metrics.csv", index=False)

    selected = select_variant(metrics)
    selected_spec = next(spec for spec in variants if spec.name == selected["variant"])
    _, selected_daily, selected_events, selected_split = evaluate_variant(
        selected_spec,
        raw_xs_events,
        raw_individual_events,
        all_dates,
        split_date,
        test_start,
        periods_per_year,
    )
    selected_daily.to_csv(OUTPUT_DIR / "selected_daily_returns.csv")
    selected_events.to_csv(OUTPUT_DIR / "selected_events.csv", index=False)
    selected_split.to_csv(OUTPUT_DIR / "selected_split_metrics.csv", index=False)

    cost_sensitivity = make_cost_sensitivity(metrics, selected)
    cost_sensitivity.to_csv(OUTPUT_DIR / "selected_cost_sensitivity.csv", index=False)
    robust_candidates = robust_1x_candidates(metrics)
    robust_candidates.to_csv(OUTPUT_DIR / "robust_1x_candidates.csv", index=False)
    cost2_survivors = cost_2x_survivors(metrics)
    cost2_survivors.to_csv(OUTPUT_DIR / "cost_2x_survivors.csv", index=False)

    plot_equity(selected_daily, split_date, test_start)
    plot_train_test(metrics)
    plot_rule_contribution(selected_events)

    write_report(
        inventory=inventory,
        raw_diagnostics=raw_diagnostics,
        metrics=metrics,
        selected=selected,
        selected_split=selected_split,
        cost_sensitivity=cost_sensitivity,
        robust_candidates=robust_candidates,
        cost2_survivors=cost2_survivors,
        split_date=split_date,
        test_start=test_start,
    )
    print(f"Selected variant: {selected['variant']}", flush=True)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
