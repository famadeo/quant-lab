"""Backtest macro-beta-neutral residual dislocations in core metals.

This is a daily prototype because the aligned CL/rates/USD 5m store only has
about one month of data. The full daily continuous set gives a defensible
2010-2024 history for the martingale-null test.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("/home/famadeo/research/databento-asset-browser/data/futures_continuous")
OUT_DIR = REPO_ROOT / "experiments" / "HYP-0028-metals-macro-residual-dislocation"
COST_PATH = (
    REPO_ROOT
    / "experiments"
    / "HYP-0014-metals-flow-filtered-residual-reversion-3y"
    / "cost_estimates.csv"
)

METALS = ["GC", "SI", "HG", "PL", "PA"]
RATES = ["ZT", "ZF", "ZN", "ZB"]
FX = ["6E", "6B", "6J", "6A", "6C"]
ALL_ROOTS = [*METALS, "CL", *RATES, *FX]

BETA_WINDOW = 252
BETA_MIN_OBS = 189
Z_WINDOW = 252
Z_MIN_OBS = 126
CORR_WINDOW = 252
CORR_MIN_OBS = 126
MD_QUANTILE_WINDOW = 504
MD_MIN_OBS = 252
PERIODS_PER_YEAR = 252
MIN_TSTAT_OBS = 2

ENTRY_Z_LEVELS = [1.5, 2.0, 2.5]
EXIT_Z_LEVELS = [0.25, 0.50]
MD_ENTRY_LABELS = ["q90", "q95"]
MD_EXIT_LABELS = ["q50", "q75"]
TOPOLOGY_EXITS = [False, True]

TRADE_OVERLAP_START = pd.Timestamp("2023-06-22", tz="UTC")
POST_2020_START = pd.Timestamp("2020-01-01", tz="UTC")


@dataclass(frozen=True)
class ActiveEvent:
    root: str
    sign: float
    neighbors: tuple[str, ...]
    entry_date: pd.Timestamp
    entry_rel_z: float
    entry_md: float


@dataclass(frozen=True)
class Variant:
    entry_z: float
    exit_z: float
    md_entry: str
    md_exit: str
    topology_exit: bool

    @property
    def name(self) -> str:
        topo = "topo_exit" if self.topology_exit else "topo_hold"
        return f"z{self.entry_z:g}_exit{self.exit_z:g}_md{self.md_entry}_norm{self.md_exit}_{topo}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_returns()
    factors = build_factors(returns)
    residuals, beta_panel = rolling_macro_residuals(returns, factors)
    residual_z = rolling_z(residuals)
    state = build_residual_state(residual_z)
    costs_bps = load_costs()

    residuals.to_parquet(OUT_DIR / "macro_residual_returns.parquet")
    beta_panel.to_parquet(OUT_DIR / "rolling_macro_betas.parquet", index=False)
    residual_z.to_parquet(OUT_DIR / "macro_residual_zscores.parquet")
    state.to_parquet(OUT_DIR / "residual_dislocation_state.parquet", index=False)

    variants = [
        Variant(entry_z, exit_z, md_entry, md_exit, topology_exit)
        for entry_z in ENTRY_Z_LEVELS
        for exit_z in EXIT_Z_LEVELS
        for md_entry in MD_ENTRY_LABELS
        for md_exit in MD_EXIT_LABELS
        for topology_exit in TOPOLOGY_EXITS
    ]
    strategy_frames: dict[str, pd.DataFrame] = {}
    event_frames: list[pd.DataFrame] = []
    metric_rows = []
    for variant in variants:
        positions_signal, events = build_positions_and_events(state, residuals, variant)
        strategy = strategy_returns(
            positions_signal,
            returns[METALS],
            residuals,
            costs_bps,
        )
        strategy["variant"] = variant.name
        strategy_frames[variant.name] = strategy.reset_index()
        events["variant"] = variant.name
        event_frames.append(events)
        metric_rows.append(metrics_for_strategy(variant.name, strategy, events))

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["net_residual_return", "residual_sharpe", "event_count"],
        ascending=[False, False, False],
    )
    events_all = pd.concat(event_frames, ignore_index=True)
    metrics.to_csv(OUT_DIR / "strategy_metrics.csv", index=False)
    events_all.to_csv(OUT_DIR / "event_log.csv", index=False)
    event_summary(events_all).to_csv(OUT_DIR / "event_summary.csv", index=False)

    split_metrics = build_split_metrics(strategy_frames, events_all)
    split_metrics.to_csv(OUT_DIR / "split_metrics.csv", index=False)

    best = metrics.iloc[0]
    best_strategy = strategy_frames[best["variant"]]
    best_strategy.to_csv(OUT_DIR / "best_strategy_returns.csv", index=False)
    plot_equity(
        best_strategy.set_index("date"),
        OUT_DIR / "best_strategy_equity.png",
        f"Best net residual variant: {best['variant']}",
    )
    plot_variant_bars(metrics, OUT_DIR / "top_variant_metrics.png")
    write_report(metrics, split_metrics, events_all, best["variant"], returns, factors)

    summary = {
        "experiment_id": "HYP-0028",
        "completed_at": datetime.now(UTC).isoformat(),
        "data_start": returns.index.min().isoformat(),
        "data_end": returns.index.max().isoformat(),
        "metals": METALS,
        "factors": ["usd", "rates_price", "CL", "leave_one_out_metals_complex"],
        "beta_window": BETA_WINDOW,
        "z_window": Z_WINDOW,
        "corr_window": CORR_WINDOW,
        "md_quantile_window": MD_QUANTILE_WINDOW,
        "costs_bps_per_side": costs_bps.to_dict(),
        "best_variant": best.to_dict(),
        "factor_start": factors.index.min().isoformat(),
        "factor_end": factors.index.max().isoformat(),
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(metrics.head(20).to_string(index=False))
    print(f"Wrote {OUT_DIR}")


def json_safe(value: Any) -> Any:  # noqa: PLR0911
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def load_returns() -> pd.DataFrame:
    parts = []
    for root in ALL_ROOTS:
        path = DATA_DIR / f"{root}.csv"
        frame = pd.read_csv(path, usecols=["date", "cont_logret"])
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        parts.append(frame.set_index("date")["cont_logret"].rename(root))
    returns = pd.concat(parts, axis=1).sort_index()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def build_factors(returns: pd.DataFrame) -> pd.DataFrame:
    factors = pd.DataFrame(index=returns.index)
    factors["usd"] = -returns[FX].mean(axis=1, skipna=False)
    factors["rates_price"] = returns[RATES].mean(axis=1, skipna=False)
    factors["CL"] = returns["CL"]
    return factors.dropna()


def rolling_macro_residuals(
    returns: pd.DataFrame, factors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual_parts = []
    beta_rows = []
    for root in METALS:
        other_metals = [candidate for candidate in METALS if candidate != root]
        root_frame = pd.concat(
            [
                returns[root].rename("metal"),
                factors,
                returns[other_metals].mean(axis=1).rename("metals_loo"),
            ],
            axis=1,
        ).dropna()
        residual = pd.Series(np.nan, index=root_frame.index, name=root)
        factor_cols = ["usd", "rates_price", "CL", "metals_loo"]
        for pos in range(BETA_MIN_OBS, len(root_frame)):
            train = root_frame.iloc[max(0, pos - BETA_WINDOW) : pos]
            if len(train) < BETA_MIN_OBS:
                continue
            x_train = train[factor_cols].to_numpy(dtype=float)
            y_train = train["metal"].to_numpy(dtype=float)
            design = np.column_stack([np.ones(len(train)), x_train])
            coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
            current_x = root_frame.iloc[pos][factor_cols].to_numpy(dtype=float)
            fitted = float(np.r_[1.0, current_x] @ coef)
            residual.iloc[pos] = float(root_frame.iloc[pos]["metal"] - fitted)
            beta_rows.append(
                {
                    "date": root_frame.index[pos],
                    "root": root,
                    "alpha": coef[0],
                    "beta_usd": coef[1],
                    "beta_rates_price": coef[2],
                    "beta_cl": coef[3],
                    "beta_metals_loo": coef[4],
                    "train_start": train.index[0],
                    "train_end": train.index[-1],
                    "train_nobs": len(train),
                }
            )
        residual_parts.append(residual)
    residuals = pd.concat(residual_parts, axis=1).dropna(how="all")
    beta_panel = pd.DataFrame(beta_rows)
    return residuals, beta_panel


def rolling_z(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.rolling(Z_WINDOW, min_periods=Z_MIN_OBS).mean().shift(1)
    std = frame.rolling(Z_WINDOW, min_periods=Z_MIN_OBS).std(ddof=1).shift(1)
    return (frame - mean) / std


def build_residual_state(residual_z: pd.DataFrame) -> pd.DataFrame:
    md_history: list[float] = []
    rows = []
    clean_z = residual_z.dropna(how="any")
    for pos in range(CORR_MIN_OBS, len(clean_z)):
        date = clean_z.index[pos]
        window = clean_z.iloc[max(0, pos - CORR_WINDOW) : pos].dropna(how="any")
        if len(window) < CORR_MIN_OBS:
            continue
        current = clean_z.iloc[pos]
        corr = window.corr().reindex(index=METALS, columns=METALS)
        neighbors = mst_neighbors(corr)
        md = mahalanobis(current.reindex(METALS), corr)
        thresholds = rolling_md_thresholds(md_history)
        row: dict[str, Any] = {
            "date": date,
            "md": md,
            **thresholds,
        }
        for root in METALS:
            root_neighbors = neighbors[root]
            row[f"{root}_neighbors"] = "|".join(root_neighbors)
            neighbor_z = current[list(root_neighbors)].mean()
            row[f"{root}_rel_z"] = float(current[root] - neighbor_z)
            row[f"{root}_z"] = float(current[root])
        rows.append(row)
        md_history.append(md)
    return pd.DataFrame(rows)


def mst_neighbors(corr: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    values = corr.to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0)
    values = np.clip(values, -0.999, 1.0)
    distance = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - values)))
    n = len(corr)
    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    edges: list[tuple[int, int]] = []
    while len(edges) < n - 1:
        best: tuple[float, int, int] | None = None
        for i in np.flatnonzero(selected):
            for j in np.flatnonzero(~selected):
                candidate = (distance[i, j], i, j)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, i, j = best
        edges.append((i, j))
        selected[j] = True
    names = list(corr.index)
    neighbors = {name: [] for name in names}
    for i, j in edges:
        neighbors[names[i]].append(names[j])
        neighbors[names[j]].append(names[i])
    return {root: tuple(sorted(items)) for root, items in neighbors.items()}


def mahalanobis(current: pd.Series, corr: pd.DataFrame) -> float:
    x = current.to_numpy(dtype=float)
    cov = corr.to_numpy(dtype=float)
    cov = np.nan_to_num(cov, nan=0.0)
    cov = np.clip(cov, -0.999, 1.0)
    cov = cov + np.eye(len(cov)) * 1e-6
    inv = np.linalg.pinv(cov)
    return float(np.sqrt(max(x @ inv @ x, 0.0)))


def rolling_md_thresholds(md_history: list[float]) -> dict[str, float]:
    if len(md_history) < MD_MIN_OBS:
        return {"md_q50": np.nan, "md_q75": np.nan, "md_q90": np.nan, "md_q95": np.nan}
    history = np.asarray(md_history[-MD_QUANTILE_WINDOW:], dtype=float)
    return {
        "md_q50": float(np.nanquantile(history, 0.50)),
        "md_q75": float(np.nanquantile(history, 0.75)),
        "md_q90": float(np.nanquantile(history, 0.90)),
        "md_q95": float(np.nanquantile(history, 0.95)),
    }


def load_costs() -> pd.Series:
    if not COST_PATH.exists():
        return pd.Series({"GC": 0.55, "SI": 1.87, "HG": 0.80, "PL": 2.56, "PA": 5.59})
    costs = pd.read_csv(COST_PATH)
    return costs.set_index("root")["per_side_cost_bps"].reindex(METALS).astype(float)


def build_positions_and_events(
    state: pd.DataFrame,
    residuals: pd.DataFrame,
    variant: Variant,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = state.sort_values("date").reset_index(drop=True)
    dates = pd.DatetimeIndex(state["date"])
    positions = np.zeros((len(state), len(METALS)), dtype=float)
    rel_z = state[[f"{root}_rel_z" for root in METALS]].to_numpy(dtype=float)
    md = state["md"].to_numpy(dtype=float)
    md_entry_threshold = state[f"md_{variant.md_entry}"].to_numpy(dtype=float)
    md_exit_threshold = state[f"md_{variant.md_exit}"].to_numpy(dtype=float)
    neighbors = {
        root_index: [
            parse_neighbors(value) for value in state[f"{root}_neighbors"].to_numpy(dtype=object)
        ]
        for root_index, root in enumerate(METALS)
    }

    active: dict[int, ActiveEvent] = {}
    events: list[dict[str, Any]] = []
    for row_index, date in enumerate(dates):
        current_md = md[row_index]
        current_md_entry = md_entry_threshold[row_index]
        current_md_exit = md_exit_threshold[row_index]
        if not np.isfinite(current_md_entry) or not np.isfinite(current_md_exit):
            continue

        for root_index in list(active):
            event = active[root_index]
            current_rel_z = rel_z[row_index, root_index]
            current_neighbors = neighbors[root_index][row_index]
            exit_event = (
                abs(current_rel_z) <= variant.exit_z
                or current_md <= current_md_exit
                or np.sign(current_rel_z) != event.sign
                or (variant.topology_exit and current_neighbors != event.neighbors)
            )
            if exit_event:
                events.append(close_event(event, date, current_rel_z, current_md, residuals))
                del active[root_index]

        for root_index, root in enumerate(METALS):
            if root_index in active:
                continue
            current_rel_z = rel_z[row_index, root_index]
            if abs(current_rel_z) >= variant.entry_z and current_md >= current_md_entry:
                root_neighbors = neighbors[root_index][row_index]
                if not root_neighbors:
                    continue
                active[root_index] = ActiveEvent(
                    root=root,
                    sign=float(np.sign(current_rel_z)),
                    neighbors=root_neighbors,
                    entry_date=date,
                    entry_rel_z=float(current_rel_z),
                    entry_md=float(current_md),
                )

        position = np.zeros(len(METALS), dtype=float)
        for event in active.values():
            event_root_index = METALS.index(event.root)
            position[event_root_index] -= event.sign
            neighbor_weight = event.sign / len(event.neighbors)
            for neighbor in event.neighbors:
                position[METALS.index(neighbor)] += neighbor_weight
        gross = float(np.abs(position).sum())
        if gross > 1.0:
            position /= gross
        positions[row_index] = position

    last_date = pd.Timestamp(dates[-1]) if len(dates) else pd.NaT
    for event in active.values():
        root_index = METALS.index(event.root)
        final_rel_z = float(rel_z[-1, root_index])
        final_md = float(md[-1])
        events.append(close_event(event, last_date, final_rel_z, final_md, residuals, forced=True))
    return pd.DataFrame(positions, index=dates, columns=METALS), pd.DataFrame(events)


def parse_neighbors(value: Any) -> tuple[str, ...]:
    if pd.isna(value) or value == "":
        return ()
    return tuple(str(value).split("|"))


def close_event(
    event: ActiveEvent,
    exit_date: pd.Timestamp,
    exit_rel_z: float,
    exit_md: float,
    residuals: pd.DataFrame,
    forced: bool = False,
) -> dict[str, Any]:
    holding = residuals.loc[(residuals.index > event.entry_date) & (residuals.index <= exit_date)]
    if holding.empty:
        residual_return = 0.0
    else:
        neighbor_return = holding[list(event.neighbors)].mean(axis=1)
        spread_return = -event.sign * holding[event.root] + event.sign * neighbor_return
        residual_return = float(spread_return.sum())
    return {
        "root": event.root,
        "sign": event.sign,
        "entry_date": event.entry_date,
        "exit_date": exit_date,
        "duration_days": int(max((exit_date - event.entry_date).days, 0)),
        "neighbors": "|".join(event.neighbors),
        "entry_rel_z": event.entry_rel_z,
        "exit_rel_z": exit_rel_z,
        "entry_md": event.entry_md,
        "exit_md": exit_md,
        "event_residual_return": residual_return,
        "forced_exit": forced,
    }


def strategy_returns(
    positions_signal: pd.DataFrame,
    raw_returns: pd.DataFrame,
    residual_returns: pd.DataFrame,
    costs_bps: pd.Series,
) -> pd.DataFrame:
    common = positions_signal.index.intersection(raw_returns.index).intersection(
        residual_returns.index
    )
    positions_signal = positions_signal.reindex(common).fillna(0.0)
    raw_returns = raw_returns.reindex(common).fillna(0.0)
    residual_returns = residual_returns.reindex(common).fillna(0.0)
    positions = positions_signal.shift(1).fillna(0.0)
    turnover = positions.diff().abs().fillna(positions.abs())
    cost = turnover.mul(costs_bps / 10_000.0, axis=1).sum(axis=1)
    gross_residual = (positions * residual_returns).sum(axis=1)
    gross_raw = (positions * raw_returns).sum(axis=1)
    frame = pd.DataFrame(
        {
            "gross_residual_return": gross_residual,
            "gross_raw_return": gross_raw,
            "cost_return": cost,
            "net_residual_return": gross_residual - cost,
            "net_raw_return": gross_raw - cost,
            "gross_exposure": positions.abs().sum(axis=1),
            "turnover": turnover.sum(axis=1),
        },
        index=common,
    )
    for root in METALS:
        frame[f"pos_{root}"] = positions[root]
    return frame


def metrics_for_strategy(
    variant_name: str,
    strategy: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, Any]:
    net = strategy["net_residual_return"]
    gross = strategy["gross_residual_return"]
    raw_net = strategy["net_raw_return"]
    event_returns = events["event_residual_return"] if not events.empty else pd.Series(dtype=float)
    return {
        "variant": variant_name,
        "gross_residual_return": gross.sum(),
        "cost_return": strategy["cost_return"].sum(),
        "net_residual_return": net.sum(),
        "net_raw_return": raw_net.sum(),
        "ann_return": net.mean() * PERIODS_PER_YEAR,
        "ann_vol": net.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR),
        "residual_sharpe": sharpe(net),
        "residual_tstat": tstat(net),
        "max_drawdown": max_drawdown(net),
        "hit_rate": float((net > 0).mean()),
        "mean_gross_exposure": strategy["gross_exposure"].mean(),
        "mean_turnover": strategy["turnover"].mean(),
        "gross_to_cost": gross.sum() / strategy["cost_return"].sum()
        if strategy["cost_return"].sum() > 0
        else np.nan,
        "event_count": len(events),
        "event_win_rate": float((event_returns > 0).mean()) if len(event_returns) else np.nan,
        "mean_event_return": event_returns.mean() if len(event_returns) else np.nan,
        "event_tstat": tstat(event_returns) if len(event_returns) > 1 else np.nan,
        "mean_duration_days": events["duration_days"].mean() if not events.empty else np.nan,
        "bars": len(strategy),
    }


def sharpe(returns: pd.Series) -> float:
    vol = returns.std(ddof=1)
    if vol <= 0 or not np.isfinite(vol):
        return np.nan
    return float(returns.mean() / vol * np.sqrt(PERIODS_PER_YEAR))


def tstat(returns: pd.Series) -> float:
    returns = returns.dropna()
    vol = returns.std(ddof=1)
    if len(returns) < MIN_TSTAT_OBS or vol <= 0 or not np.isfinite(vol):
        return np.nan
    return float(returns.mean() / vol * np.sqrt(len(returns)))


def max_drawdown(returns: pd.Series) -> float:
    equity = returns.fillna(0.0).cumsum()
    drawdown = equity - equity.cummax()
    return float(drawdown.min())


def event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    grouped = events.groupby(["variant", "root", "sign"], dropna=False)
    rows = []
    for keys, group in grouped:
        returns = group["event_residual_return"]
        rows.append(
            {
                "variant": keys[0],
                "root": keys[1],
                "sign": keys[2],
                "events": len(group),
                "mean_event_return": returns.mean(),
                "median_event_return": returns.median(),
                "event_tstat": tstat(returns),
                "win_rate": float((returns > 0).mean()),
                "mean_duration_days": group["duration_days"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_event_return", "event_tstat"], ascending=False)


def build_split_metrics(
    strategy_frames: dict[str, pd.DataFrame], events: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for variant, strategy_frame in strategy_frames.items():
        frame = strategy_frame.copy()
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        split_defs = {
            "full": frame,
            "pre_2020": frame[frame["date"] < POST_2020_START],
            "post_2020": frame[frame["date"] >= POST_2020_START],
            "trade_overlap": frame[frame["date"] >= TRADE_OVERLAP_START],
        }
        for split, split_frame in split_defs.items():
            split_events = events[
                (events["variant"] == variant)
                & (pd.to_datetime(events["entry_date"], utc=True).isin(split_frame["date"]))
            ]
            if split_frame.empty:
                continue
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "start": split_frame["date"].min(),
                    "end": split_frame["date"].max(),
                    "bars": len(split_frame),
                    "net_residual_return": split_frame["net_residual_return"].sum(),
                    "net_raw_return": split_frame["net_raw_return"].sum(),
                    "cost_return": split_frame["cost_return"].sum(),
                    "residual_sharpe": sharpe(split_frame["net_residual_return"]),
                    "residual_tstat": tstat(split_frame["net_residual_return"]),
                    "max_drawdown": max_drawdown(split_frame["net_residual_return"]),
                    "events": len(split_events),
                }
            )
    return pd.DataFrame(rows)


def plot_equity(strategy: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    equity = strategy["net_residual_return"].fillna(0.0).cumsum()
    raw_equity = strategy["net_raw_return"].fillna(0.0).cumsum()
    axes[0].plot(equity.index, equity, label="macro-residual net", color="#2f6f9f")
    axes[0].plot(raw_equity.index, raw_equity, label="raw metal-basket net", color="#8d5a2b")
    axes[0].set_title(title)
    axes[0].set_ylabel("Cumulative log return")
    axes[0].legend()
    drawdown = equity - equity.cummax()
    axes[1].fill_between(drawdown.index, drawdown, 0.0, color="#9f3d3d", alpha=0.35)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_variant_bars(metrics: pd.DataFrame, out_path: Path) -> None:
    top = metrics.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top["variant"], top["net_residual_return"], color="#2f6f9f")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top variants by net macro-residual return")
    ax.set_xlabel("Cumulative net residual log return")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(
    metrics: pd.DataFrame,
    split_metrics: pd.DataFrame,
    events: pd.DataFrame,
    best_variant: str,
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> None:
    best_splits = split_metrics[split_metrics["variant"] == best_variant]
    best_events = events[events["variant"] == best_variant]
    report = [
        "# HYP-0028 Metals Macro-Residual Dislocation Strategy",
        "",
        f"Completed at `{datetime.now(UTC).isoformat()}`.",
        "",
        "## Design",
        "",
        "- Frequency: daily continuous futures, because aligned 5m macro factors "
        "only cover about one month.",
        "- Residual model: rolling lagged OLS per metal on USD, rates-price, CL, "
        "and leave-one-out metals complex.",
        "- Dislocation state: residual z-scores, rolling residual-correlation MST "
        "neighbors, and residual-cloud Mahalanobis distance.",
        "- Entry: root-vs-MST-neighbor residual spread exceeds z threshold and "
        "residual-cloud MD exceeds rolling threshold.",
        "- Exit: spread normalizes, MD returns to normal, sign flips, or optionally "
        "MST topology changes. No fixed-time exit.",
        "- Execution: signal at close `t`; position earns return at `t+1`; "
        "metal costs charged on turnover.",
        "- PnL focus: `net_residual_return` is the macro-hedged residual alpha test. "
        "`net_raw_return` is the unhedged metal-basket implementation check.",
        "",
        "## Coverage",
        "",
        f"- Return/factor span: `{returns.index.min().date()}` to `{returns.index.max().date()}`.",
        f"- Factor panel span: `{factors.index.min().date()}` to `{factors.index.max().date()}`.",
        "- Explicit carry-curve adjustment is not included in this daily prototype; "
        "continuous roll-adjusted futures returns are used.",
        "- Macro hedge costs are not included, so residual PnL is optimistic relative "
        "to a fully executable hedge package.",
        "",
        "## Best Variant",
        "",
        metrics.loc[
            metrics["variant"].eq(best_variant),
            [
                "variant",
                "net_residual_return",
                "cost_return",
                "net_raw_return",
                "residual_sharpe",
                "residual_tstat",
                "max_drawdown",
                "event_count",
                "mean_event_return",
                "event_tstat",
            ],
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Best Variant Splits",
        "",
        best_splits[
            [
                "split",
                "start",
                "end",
                "net_residual_return",
                "net_raw_return",
                "cost_return",
                "residual_sharpe",
                "residual_tstat",
                "events",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Metrics",
        "",
        metrics.head(12)[
            [
                "variant",
                "net_residual_return",
                "cost_return",
                "net_raw_return",
                "residual_sharpe",
                "residual_tstat",
                "event_count",
                "event_tstat",
                "mean_duration_days",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Event Direction Notes",
        "",
        event_summary(best_events).head(12).to_markdown(index=False, floatfmt=".4f")
        if not best_events.empty
        else "No events.",
        "",
        "## Files",
        "",
        "- `strategy_metrics.csv`",
        "- `split_metrics.csv`",
        "- `event_log.csv`",
        "- `event_summary.csv`",
        "- `macro_residual_returns.parquet`",
        "- `rolling_macro_betas.parquet`",
        "- `residual_dislocation_state.parquet`",
        "- `best_strategy_returns.csv`",
        "- `best_strategy_equity.png`",
        "- `top_variant_metrics.png`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
