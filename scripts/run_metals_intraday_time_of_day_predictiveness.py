from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
CONTINUOUS_DIR = Path(
    "/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/continuous"
)
OUTPUT_DIR = Path("experiments/HYP-0034-metals-intraday-time-of-day-predictiveness")

SIGNAL_WINDOWS = [5, 15, 30, 60]
FORWARD_HORIZONS = [5, 15, 30, 60, 120]
BIN_MINUTES = 30
MIN_TSTAT_OBS = 2
MIN_CORR_OBS = 3
MIN_OBS_PER_BIN = 5
MIN_DAYS = 250
TRAIN_FRACTION = 0.70
TOP_Q = 0.80
BOTTOM_Q = 0.20


def safe_tstat(values: pd.Series) -> float:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < MIN_TSTAT_OBS:
        return np.nan
    std = clean.std(ddof=1)
    if not np.isfinite(std) or std <= 0:
        return np.nan
    return float(clean.mean() / std * math.sqrt(len(clean)))


def normal_p_from_t(t_value: float) -> float:
    if not np.isfinite(t_value):
        return np.nan
    return math.erfc(abs(t_value) / math.sqrt(2.0))


def bh_qvalues(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return pd.Series(q, index=p_values.index)
    valid_idx = np.where(valid)[0]
    order = valid_idx[np.argsort(p[valid])]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.minimum(ranked, 1.0)
    return pd.Series(q, index=p_values.index)


def corr_tstat(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    frame = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < MIN_CORR_OBS:
        return np.nan, np.nan, np.nan
    x_values = frame.iloc[:, 0].to_numpy(dtype=float)
    y_values = frame.iloc[:, 1].to_numpy(dtype=float)
    x_demeaned = x_values - x_values.mean()
    y_demeaned = y_values - y_values.mean()
    denominator = math.sqrt(float(np.sum(x_demeaned**2) * np.sum(y_demeaned**2)))
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan, np.nan, np.nan
    corr = float(np.sum(x_demeaned * y_demeaned) / denominator)
    corr = min(max(corr, -1.0), 1.0)
    if not np.isfinite(corr) or abs(corr) >= 1.0:
        return corr, np.nan, np.nan
    t_value = corr * math.sqrt((len(frame) - 2) / max(1e-12, 1.0 - corr * corr))
    return corr, float(t_value), normal_p_from_t(t_value)


def split_date(dates: pd.Series) -> pd.Timestamp:
    unique_dates = pd.Series(pd.to_datetime(dates).dropna().unique()).sort_values().reset_index(
        drop=True
    )
    if unique_dates.empty:
        return pd.Timestamp.max.tz_localize("UTC")
    split_idx = min(max(int(len(unique_dates) * TRAIN_FRACTION), 1), len(unique_dates) - 1)
    return pd.Timestamp(unique_dates.iloc[split_idx])


def time_label(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def load_root_daily_bins(root: str) -> pd.DataFrame:
    path = CONTINUOUS_DIR / f"{root}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    raw = (
        pl.scan_parquet(path)
        .select("ts", "cont_logprice", "volume", "is_roll")
        .collect()
        .to_pandas()
    )
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
    raw = raw.sort_values("ts").replace([np.inf, -np.inf], np.nan)
    roll_dates = set(raw.loc[raw["is_roll"].fillna(False), "ts"].dt.normalize())

    full_index = pd.date_range(raw["ts"].min(), raw["ts"].max(), freq="1min", tz="UTC")
    minute = raw.set_index("ts")[["cont_logprice", "volume"]].reindex(full_index)
    minute["observed"] = minute["cont_logprice"].notna()
    minute["volume"] = minute["volume"].fillna(0.0)
    minute["logp"] = minute["cont_logprice"].ffill()
    minute = minute.drop(columns=["cont_logprice"])
    minute["date"] = minute.index.normalize()
    if roll_dates:
        minute = minute.loc[~minute["date"].isin(roll_dates)].copy()
    minute = minute.dropna(subset=["logp"])
    minute["minute_of_day"] = minute.index.hour * 60 + minute.index.minute
    minute["time_bin"] = (minute["minute_of_day"] // BIN_MINUTES) * BIN_MINUTES
    minute["decision_mark"] = minute["minute_of_day"] % BIN_MINUTES == 0
    minute["obs_bars"] = minute.groupby("date", sort=False)["observed"].transform(
        lambda values: values.rolling(BIN_MINUTES, min_periods=1).sum()
    )
    minute["volume_window"] = minute.groupby("date", sort=False)["volume"].transform(
        lambda values: values.rolling(BIN_MINUTES, min_periods=1).sum()
    )

    grouped = minute.groupby("date", sort=False)["logp"]
    for window in SIGNAL_WINDOWS:
        minute[f"sig_{window}m"] = minute["logp"] - grouped.shift(window)
    for horizon in FORWARD_HORIZONS:
        minute[f"fwd_{horizon}m"] = grouped.shift(-horizon) - minute["logp"]
    minute["fwd_rest"] = grouped.transform("last") - minute["logp"]

    value_cols = [
        "date",
        "time_bin",
        "obs_bars",
        "volume_window",
        *[f"sig_{window}m" for window in SIGNAL_WINDOWS],
        *[f"fwd_{horizon}m" for horizon in FORWARD_HORIZONS],
        "fwd_rest",
    ]
    bins = minute.loc[minute["decision_mark"], value_cols].reset_index(drop=True)
    bins = bins[bins["obs_bars"] >= MIN_OBS_PER_BIN].copy()
    bins = bins.rename(columns={"volume_window": "volume"})
    bins["root"] = root
    bins["time_utc"] = bins["time_bin"].map(time_label)
    bins["panel_version"] = "decision_mark_v2"
    return bins


def build_daily_bin_panel() -> pd.DataFrame:
    frames = []
    inventory = []
    for root in ROOTS:
        print(f"Building daily intraday bins for {root}", flush=True)
        bins = load_root_daily_bins(root)
        frames.append(bins)
        inventory.append(
            {
                "root": root,
                "rows": len(bins),
                "days": bins["date"].nunique(),
                "first_date": bins["date"].min(),
                "last_date": bins["date"].max(),
                "median_bins_per_day": bins.groupby("date").size().median(),
                "median_obs_per_decision_mark": bins["obs_bars"].median(),
            }
        )
    panel = pd.concat(frames, ignore_index=True).sort_values(["root", "date", "time_bin"])
    pd.DataFrame(inventory).to_csv(OUTPUT_DIR / "data_inventory.csv", index=False)
    return panel


def summarize_values(values: pd.Series) -> dict[str, float]:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n": len(clean),
        "mean": float(clean.mean()) if len(clean) else np.nan,
        "tstat": safe_tstat(clean),
        "p_value": normal_p_from_t(safe_tstat(clean)),
    }


def summarize_train_test(
    data: pd.DataFrame, value_col: str, date_col: str = "date"
) -> dict[str, float]:
    split = split_date(data[date_col])
    train = data[data[date_col] < split][value_col]
    test = data[data[date_col] >= split][value_col]
    train_stats = summarize_values(train)
    test_stats = summarize_values(test)
    return {
        "split_date": split,
        "train_n": train_stats["n"],
        "train_mean": train_stats["mean"],
        "train_tstat": train_stats["tstat"],
        "test_n": test_stats["n"],
        "test_mean": test_stats["mean"],
        "test_tstat": test_stats["tstat"],
    }


def unconditional_drift(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for root, root_frame in panel.groupby("root", sort=False):
        for time_bin, bin_frame in root_frame.groupby("time_bin", sort=True):
            for horizon in [*FORWARD_HORIZONS, "rest"]:
                fwd_col = f"fwd_{horizon}m" if isinstance(horizon, int) else "fwd_rest"
                values = bin_frame[fwd_col]
                stats = summarize_values(values)
                if stats["n"] < MIN_DAYS:
                    continue
                split_stats = summarize_train_test(bin_frame, fwd_col)
                rows.append(
                    {
                        "root": root,
                        "time_bin": time_bin,
                        "time_utc": time_label(int(time_bin)),
                        "horizon": horizon,
                        **stats,
                        **split_stats,
                    }
                )
    out = pd.DataFrame(rows)
    out["q_value"] = bh_qvalues(out["p_value"]) if not out.empty else np.nan
    return out.sort_values(["q_value", "p_value", "root", "time_bin"])


def quintile_spread(frame: pd.DataFrame, signal_col: str, fwd_col: str) -> pd.Series:
    data = frame[["date", signal_col, fwd_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < MIN_DAYS:
        return pd.Series(dtype=float)
    lo = data[signal_col].quantile(BOTTOM_Q)
    hi = data[signal_col].quantile(TOP_Q)
    low = data.loc[data[signal_col] <= lo, fwd_col].mean()
    high = data.loc[data[signal_col] >= hi, fwd_col].mean()
    if not np.isfinite(high) or not np.isfinite(low):
        return pd.Series(dtype=float)
    # Build a daily pseudo-return: top-quintile days receive +fwd, bottom-quintile days -fwd.
    top = data.loc[data[signal_col] >= hi, ["date", fwd_col]].copy()
    bot = data.loc[data[signal_col] <= lo, ["date", fwd_col]].copy()
    top["value"] = top[fwd_col]
    bot["value"] = -bot[fwd_col]
    return pd.concat([top[["date", "value"]], bot[["date", "value"]]]).groupby("date")[
        "value"
    ].mean()


def individual_predictiveness(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (root, time_bin), group in panel.groupby(["root", "time_bin"], sort=True):
        for window in SIGNAL_WINDOWS:
            signal_col = f"sig_{window}m"
            for horizon in [*FORWARD_HORIZONS, "rest"]:
                fwd_col = f"fwd_{horizon}m" if isinstance(horizon, int) else "fwd_rest"
                data = group[["date", signal_col, fwd_col]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if len(data) < MIN_DAYS:
                    continue
                corr, corr_t, corr_p = corr_tstat(data[signal_col], data[fwd_col])
                signed = np.sign(data[signal_col]) * data[fwd_col]
                signed_stats = summarize_values(signed)
                q_spread = quintile_spread(data, signal_col, fwd_col)
                q_stats = summarize_values(q_spread)
                split_stats = summarize_train_test(
                    pd.DataFrame({"date": q_spread.index, "q_value_return": q_spread.values}),
                    "q_value_return",
                )
                rows.append(
                    {
                        "root": root,
                        "time_bin": time_bin,
                        "time_utc": time_label(int(time_bin)),
                        "signal_window": window,
                        "horizon": horizon,
                        "n": len(data),
                        "ic": corr,
                        "ic_tstat": corr_t,
                        "ic_p_value": corr_p,
                        "signed_mean": signed_stats["mean"],
                        "signed_tstat": signed_stats["tstat"],
                        "signed_p_value": signed_stats["p_value"],
                        "quintile_spread_mean": q_stats["mean"],
                        "quintile_spread_tstat": q_stats["tstat"],
                        "quintile_spread_p_value": q_stats["p_value"],
                        **split_stats,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["ic_q_value"] = bh_qvalues(out["ic_p_value"])
    out["signed_q_value"] = bh_qvalues(out["signed_p_value"])
    out["quintile_spread_q_value"] = bh_qvalues(out["quintile_spread_p_value"])
    return out.sort_values(
        ["quintile_spread_q_value", "quintile_spread_p_value", "root", "time_bin"]
    )


def cross_sectional_predictiveness(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for time_bin, time_frame in panel.groupby("time_bin", sort=True):
        for window in SIGNAL_WINDOWS:
            signal_col = f"sig_{window}m"
            for horizon in [*FORWARD_HORIZONS, "rest"]:
                fwd_col = f"fwd_{horizon}m" if isinstance(horizon, int) else "fwd_rest"
                signal = time_frame.pivot_table(
                    index="date", columns="root", values=signal_col, aggfunc="mean"
                ).reindex(columns=ROOTS)
                fwd = time_frame.pivot_table(
                    index="date", columns="root", values=fwd_col, aggfunc="mean"
                ).reindex(index=signal.index, columns=ROOTS)
                signal_values = signal.to_numpy(dtype=float)
                fwd_values = fwd.to_numpy(dtype=float)
                valid = np.isfinite(signal_values) & np.isfinite(fwd_values)
                valid_counts = valid.sum(axis=1)
                usable = valid_counts >= MIN_CORR_OBS
                if usable.sum() < MIN_DAYS:
                    continue

                masked_signal = np.where(valid, signal_values, np.nan)
                masked_fwd = np.where(valid, fwd_values, np.nan)
                winner_idx = np.nanargmax(masked_signal[usable], axis=1)
                loser_idx = np.nanargmin(masked_signal[usable], axis=1)
                usable_fwd = masked_fwd[usable]
                row_num = np.arange(usable_fwd.shape[0])
                ls = pd.DataFrame(
                    {
                        "date": signal.index[usable],
                        "momentum_ls_return": usable_fwd[row_num, winner_idx]
                        - usable_fwd[row_num, loser_idx],
                    }
                )

                signal_mean = np.nanmean(masked_signal[usable], axis=1, keepdims=True)
                fwd_mean = np.nanmean(masked_fwd[usable], axis=1, keepdims=True)
                signal_demeaned = np.where(
                    np.isfinite(masked_signal[usable]),
                    masked_signal[usable] - signal_mean,
                    np.nan,
                )
                fwd_demeaned = np.where(
                    np.isfinite(masked_fwd[usable]),
                    masked_fwd[usable] - fwd_mean,
                    np.nan,
                )
                numerator = np.nansum(signal_demeaned * fwd_demeaned, axis=1)
                denominator = np.sqrt(
                    np.nansum(signal_demeaned**2, axis=1) * np.nansum(fwd_demeaned**2, axis=1)
                )
                daily_ic = np.divide(
                    numerator,
                    denominator,
                    out=np.full_like(numerator, np.nan, dtype=float),
                    where=denominator > 0,
                )
                ic = pd.DataFrame({"date": signal.index[usable], "daily_ic": daily_ic}).dropna()
                if len(ls) < MIN_DAYS:
                    continue
                ls_stats = summarize_values(ls["momentum_ls_return"])
                ic_stats = summarize_values(ic["daily_ic"]) if not ic.empty else {}
                split_stats = summarize_train_test(ls, "momentum_ls_return")
                rows.append(
                    {
                        "time_bin": time_bin,
                        "time_utc": time_label(int(time_bin)),
                        "signal_window": window,
                        "horizon": horizon,
                        "n_days": len(ls),
                        "momentum_ls_mean": ls_stats["mean"],
                        "momentum_ls_tstat": ls_stats["tstat"],
                        "momentum_ls_p_value": ls_stats["p_value"],
                        "mean_daily_ic": ic_stats.get("mean", np.nan),
                        "daily_ic_tstat": ic_stats.get("tstat", np.nan),
                        "daily_ic_p_value": ic_stats.get("p_value", np.nan),
                        **split_stats,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["momentum_ls_q_value"] = bh_qvalues(out["momentum_ls_p_value"])
    out["daily_ic_q_value"] = bh_qvalues(out["daily_ic_p_value"])
    return out.sort_values(["momentum_ls_q_value", "momentum_ls_p_value", "time_bin"])


def plot_unconditional(uncond: pd.DataFrame, output_path: Path) -> None:
    data = uncond[uncond["horizon"].eq(60)].copy()
    if data.empty:
        return
    pivot = data.pivot_table(index="root", columns="time_utc", values="tstat", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(16, 4.5), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    tick_positions = np.arange(0, len(pivot.columns), 4)
    ax.set_xticks(tick_positions, labels=[pivot.columns[i] for i in tick_positions], rotation=45)
    ax.set_title("Unconditional 60-minute forward-return t-stat by UTC time")
    fig.colorbar(im, ax=ax, label="t-stat")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cross_sectional(xs: pd.DataFrame, output_path: Path) -> None:
    data = xs[(xs["signal_window"].eq(30)) & (xs["horizon"].isin(FORWARD_HORIZONS))].copy()
    if data.empty:
        return
    pivot = data.pivot_table(
        index="horizon", columns="time_utc", values="momentum_ls_tstat", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(16, 4.5), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    tick_positions = np.arange(0, len(pivot.columns), 4)
    ax.set_xticks(tick_positions, labels=[pivot.columns[i] for i in tick_positions], rotation=45)
    ax.set_title("Cross-sectional 30-minute signal momentum t-stat by UTC time")
    ax.set_ylabel("Forward horizon minutes")
    fig.colorbar(im, ax=ax, label="t-stat")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_individual_top(individual: pd.DataFrame, output_path: Path) -> None:
    if individual.empty:
        return
    rows = []
    for (root, time_utc), group in individual.groupby(["root", "time_utc"], sort=False):
        best = group.iloc[group["quintile_spread_tstat"].abs().argmax()]
        rows.append(
            {
                "root": root,
                "time_utc": time_utc,
                "best_abs_tstat": best["quintile_spread_tstat"],
            }
        )
    data = pd.DataFrame(rows)
    pivot = data.pivot_table(index="root", columns="time_utc", values="best_abs_tstat")
    fig, ax = plt.subplots(figsize=(16, 4.5), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    tick_positions = np.arange(0, len(pivot.columns), 4)
    ax.set_xticks(tick_positions, labels=[pivot.columns[i] for i in tick_positions], rotation=45)
    ax.set_title("Best individual prior-return quintile-spread t-stat by UTC time")
    fig.colorbar(im, ax=ax, label="t-stat")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def stable_rows(frame: pd.DataFrame, stat_col: str, mean_col: str, q_col: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    data = frame.copy()
    same_sign = np.sign(data[mean_col]) == np.sign(data["test_mean"])
    return data[same_sign & data[q_col].lt(0.10) & data[stat_col].abs().ge(2.0)].copy()


def write_report(
    *,
    panel: pd.DataFrame,
    uncond: pd.DataFrame,
    individual: pd.DataFrame,
    xs: pd.DataFrame,
    stable_individual: pd.DataFrame,
    stable_xs: pd.DataFrame,
) -> None:
    inventory = pd.read_csv(OUTPUT_DIR / "data_inventory.csv")
    top_ind_cols = [
        "root",
        "time_utc",
        "signal_window",
        "horizon",
        "n",
        "ic",
        "ic_tstat",
        "quintile_spread_mean",
        "quintile_spread_tstat",
        "quintile_spread_q_value",
        "train_mean",
        "train_tstat",
        "test_mean",
        "test_tstat",
    ]
    top_xs_cols = [
        "time_utc",
        "signal_window",
        "horizon",
        "n_days",
        "momentum_ls_mean",
        "momentum_ls_tstat",
        "momentum_ls_q_value",
        "mean_daily_ic",
        "daily_ic_tstat",
        "train_mean",
        "train_tstat",
        "test_mean",
        "test_tstat",
    ]
    top_uncond_cols = [
        "root",
        "time_utc",
        "horizon",
        "n",
        "mean",
        "tstat",
        "q_value",
        "train_mean",
        "train_tstat",
        "test_mean",
        "test_tstat",
    ]
    lines = [
        "# HYP-0034 Metals Intraday Time-of-Day Predictiveness",
        "",
        "## Design",
        "",
        "- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.",
        "- Data: 1-minute return-spliced continuous futures from 2016-06-22 through 2026-06-21.",
        "- Roll bars are removed. Returns are evaluated within the same UTC calendar day.",
        f"- Time-of-day unit: fixed {BIN_MINUTES}-minute UTC decision marks with at least "
        f"{MIN_OBS_PER_BIN} observed bars in the trailing {BIN_MINUTES} minutes.",
        "- Individual tests: prior 5/15/30/60 minute return known at the decision mark versus "
        "forward "
        "5/15/30/60/120 minute and rest-of-day return.",
        "- Cross-sectional tests: at each day/time decision mark, long the metal with highest "
        "prior return and short the metal with lowest prior return, then measure forward spread.",
        "- Significance uses daily decision-mark t-statistics and Benjamini-Hochberg q-values. "
        "Train/test split is chronological 70/30 by date.",
        "",
        "## Data Inventory",
        "",
        inventory.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Top Unconditional Time-of-Day Drifts",
        "",
        uncond.head(20)[top_uncond_cols].to_markdown(index=False, floatfmt=".6f")
        if not uncond.empty
        else "No unconditional rows.",
        "",
        "## Top Individual Conditional Patterns",
        "",
        individual.head(25)[top_ind_cols].to_markdown(index=False, floatfmt=".6f")
        if not individual.empty
        else "No individual rows.",
        "",
        "## Stable Individual Patterns",
        "",
        stable_individual.head(25)[top_ind_cols].to_markdown(index=False, floatfmt=".6f")
        if not stable_individual.empty
        else "No individual pattern passed the stability filter.",
        "",
        "## Top Cross-Sectional Patterns",
        "",
        xs.head(25)[top_xs_cols].to_markdown(index=False, floatfmt=".6f")
        if not xs.empty
        else "No cross-sectional rows.",
        "",
        "## Stable Cross-Sectional Patterns",
        "",
        stable_xs.head(25)[top_xs_cols].to_markdown(index=False, floatfmt=".6f")
        if not stable_xs.empty
        else "No cross-sectional pattern passed the stability filter.",
        "",
        "## Interpretation",
        "",
        "This is a discovery pass, not a tradable strategy. Any candidate that survives here "
        "still needs explicit execution costs, latency assumptions, non-overlapping event tests, "
        "and an out-of-sample implementation backtest.",
        "",
        "## Files",
        "",
        "- `daily_time_bin_panel.parquet`",
        "- `unconditional_time_of_day_drift.csv`",
        "- `individual_time_of_day_predictiveness.csv`",
        "- `cross_sectional_time_of_day_predictiveness.csv`",
        "- `stable_individual_patterns.csv`",
        "- `stable_cross_sectional_patterns.csv`",
        "- `data_inventory.csv`",
        "- `unconditional_60m_tstat_heatmap.png`",
        "- `individual_best_tstat_heatmap.png`",
        "- `cross_sectional_30m_tstat_heatmap.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_path = OUTPUT_DIR / "daily_time_bin_panel.parquet"
    if panel_path.exists():
        panel = pd.read_parquet(panel_path)
        if "panel_version" not in panel.columns or not panel["panel_version"].eq(
            "decision_mark_v2"
        ).all():
            panel = build_daily_bin_panel()
            panel.to_parquet(panel_path, index=False)
    else:
        panel = build_daily_bin_panel()
        panel.to_parquet(panel_path, index=False)
    print(f"Daily bin panel rows: {len(panel):,}", flush=True)

    uncond = unconditional_drift(panel)
    individual = individual_predictiveness(panel)
    xs = cross_sectional_predictiveness(panel)

    stable_individual = stable_rows(
        individual,
        "quintile_spread_tstat",
        "quintile_spread_mean",
        "quintile_spread_q_value",
    )
    stable_xs = stable_rows(
        xs,
        "momentum_ls_tstat",
        "momentum_ls_mean",
        "momentum_ls_q_value",
    )

    uncond.to_csv(OUTPUT_DIR / "unconditional_time_of_day_drift.csv", index=False)
    individual.to_csv(OUTPUT_DIR / "individual_time_of_day_predictiveness.csv", index=False)
    xs.to_csv(OUTPUT_DIR / "cross_sectional_time_of_day_predictiveness.csv", index=False)
    stable_individual.to_csv(OUTPUT_DIR / "stable_individual_patterns.csv", index=False)
    stable_xs.to_csv(OUTPUT_DIR / "stable_cross_sectional_patterns.csv", index=False)

    plot_unconditional(uncond, OUTPUT_DIR / "unconditional_60m_tstat_heatmap.png")
    plot_individual_top(individual, OUTPUT_DIR / "individual_best_tstat_heatmap.png")
    plot_cross_sectional(xs, OUTPUT_DIR / "cross_sectional_30m_tstat_heatmap.png")

    write_report(
        panel=panel,
        uncond=uncond,
        individual=individual,
        xs=xs,
        stable_individual=stable_individual,
        stable_xs=stable_xs,
    )
    print("Top stable individual patterns:")
    print(stable_individual.head(10).round(6).to_string(index=False))
    print("Top stable cross-sectional patterns:")
    print(stable_xs.head(10).round(6).to_string(index=False))
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
