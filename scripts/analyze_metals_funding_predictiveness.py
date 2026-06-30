from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "experiments" / "HYP-0037-metals-funding-vs-realized-returns"
HORIZON_PATH = SOURCE_DIR / "forward_horizon_accounting.parquet"
CUMULATIVE_PATH = SOURCE_DIR / "cumulative_hourly_accounting.parquet"
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0038-metals-funding-predictiveness"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
TARGET_MONTHS = [1, 3, 6]
HORIZONS = ["1h", "4h", "1d", "3d", "1w", "1m"]
SIGNALS = [
    "funding_pct_ann",
    "funding_z_126d",
    "funding_change_24h_pct_ann",
    "funding_z_change_24h",
    "abs_funding_z_126d",
]

MIN_OBSERVATIONS = 250
HAC_MAXLAGS_CAP = 168
REPORT_TARGET_MONTHS = 3


def load_signal_panel() -> pd.DataFrame:
    if not CUMULATIVE_PATH.exists():
        raise FileNotFoundError(CUMULATIVE_PATH)
    panel = pd.read_parquet(
        CUMULATIVE_PATH,
        columns=[
            "root",
            "target_months",
            "ts",
            "funding_rate",
            "funding_pct_ann",
            "funding_z_126d",
            "tenor_months",
        ],
    )
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel = panel.sort_values(["root", "target_months", "ts"])
    grouped = panel.groupby(["root", "target_months"], sort=False)
    panel["funding_change_1h_pct_ann"] = grouped["funding_pct_ann"].diff(1)
    panel["funding_change_24h_pct_ann"] = grouped["funding_pct_ann"].diff(24)
    panel["funding_change_5d_pct_ann"] = grouped["funding_pct_ann"].diff(120)
    panel["funding_z_change_24h"] = grouped["funding_z_126d"].diff(24)
    panel["abs_funding_z_126d"] = panel["funding_z_126d"].abs()
    panel["curve_state"] = np.where(
        panel["funding_rate"].gt(0.0),
        "contango",
        np.where(panel["funding_rate"].lt(0.0), "backwardation", "flat"),
    )
    return panel


def load_horizon_panel() -> pd.DataFrame:
    if not HORIZON_PATH.exists():
        raise FileNotFoundError(HORIZON_PATH)
    horizons = pd.read_parquet(
        HORIZON_PATH,
        columns=[
            "root",
            "target_months",
            "horizon",
            "horizon_bars",
            "ts",
            "forward_log_return",
            "path_funding_paid_log",
            "excess_after_path_funding_log",
        ],
    )
    horizons["ts"] = pd.to_datetime(horizons["ts"], utc=True)
    return horizons.sort_values(["root", "target_months", "horizon", "ts"])


def hac_ols_stats(x: pd.Series, y: pd.Series, maxlags: int) -> dict[str, float]:
    data = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    data = data[np.isfinite(data["x"]) & np.isfinite(data["y"])]
    if len(data) < MIN_OBSERVATIONS or data["x"].std(ddof=1) == 0:
        return {"beta": np.nan, "t": np.nan, "p": np.nan, "intercept": np.nan}
    x_std = (data["x"] - data["x"].mean()) / data["x"].std(ddof=1)
    result = sm.OLS(data["y"], sm.add_constant(x_std)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags},
    )
    return {
        "intercept": float(result.params["const"]),
        "beta": float(result.params["x"]),
        "t": float(result.tvalues["x"]),
        "p": float(result.pvalues["x"]),
    }


def summarize_signal(
    *,
    root: str,
    target_months: int,
    horizon: str,
    horizon_bars: int,
    signal: str,
    data: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    aligned = data[[signal, "excess_after_path_funding_log"]].dropna()
    aligned = aligned[
        np.isfinite(aligned[signal])
        & np.isfinite(aligned["excess_after_path_funding_log"])
    ]
    if len(aligned) < MIN_OBSERVATIONS or aligned[signal].std(ddof=1) == 0:
        return (
            {
                "root": root,
                "target_months": target_months,
                "horizon": horizon,
                "signal": signal,
                "n": len(aligned),
            },
            pd.DataFrame(),
        )

    y = aligned["excess_after_path_funding_log"]
    x = aligned[signal]
    maxlags = min(max(1, horizon_bars - 1), HAC_MAXLAGS_CAP)
    ols = hac_ols_stats(x, y, maxlags=maxlags)
    low_cut = x.quantile(0.10)
    high_cut = x.quantile(0.90)
    low = aligned[x <= low_cut]["excess_after_path_funding_log"]
    high = aligned[x >= high_cut]["excess_after_path_funding_log"]
    middle = aligned[(x > low_cut) & (x < high_cut)]["excess_after_path_funding_log"]

    try:
        decile_index = pd.qcut(x, 10, labels=False, duplicates="drop")
        deciles = (
            aligned.assign(decile=decile_index)
            .dropna(subset=["decile"])
            .groupby("decile", as_index=False)
            .agg(
                n=("excess_after_path_funding_log", "size"),
                mean_excess_bp=("excess_after_path_funding_log", lambda v: v.mean() * 10_000.0),
                median_excess_bp=(
                    "excess_after_path_funding_log",
                    lambda v: v.median() * 10_000.0,
                ),
                mean_signal=(signal, "mean"),
            )
        )
        deciles["decile"] = deciles["decile"].astype(int) + 1
        deciles.insert(0, "signal", signal)
        deciles.insert(0, "horizon", horizon)
        deciles.insert(0, "target_months", target_months)
        deciles.insert(0, "root", root)
    except ValueError:
        deciles = pd.DataFrame()

    return (
        {
            "root": root,
            "target_months": target_months,
            "horizon": horizon,
            "horizon_bars": horizon_bars,
            "signal": signal,
            "n": len(aligned),
            "signal_mean": x.mean(),
            "signal_std": x.std(ddof=1),
            "pearson_corr": x.corr(y, method="pearson"),
            "spearman_corr": stats.spearmanr(x, y).statistic,
            "beta_bp_per_1sd_signal": ols["beta"] * 10_000.0,
            "hac_t": ols["t"],
            "hac_p": ols["p"],
            "low_decile_excess_bp": low.mean() * 10_000.0,
            "middle_excess_bp": middle.mean() * 10_000.0,
            "high_decile_excess_bp": high.mean() * 10_000.0,
            "high_minus_low_excess_bp": (high.mean() - low.mean()) * 10_000.0,
            "high_positive_fraction": float((high > 0).mean()),
            "low_positive_fraction": float((low > 0).mean()),
        },
        deciles,
    )


def summarize_state(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (root, target, horizon), group in data.groupby(
        ["root", "target_months", "horizon"], sort=True
    ):
        by_state = group.groupby("curve_state")["excess_after_path_funding_log"]
        contango = (
            by_state.get_group("contango")
            if "contango" in by_state.groups
            else pd.Series(dtype=float)
        )
        backwardation = (
            by_state.get_group("backwardation")
            if "backwardation" in by_state.groups
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "root": root,
                "target_months": target,
                "horizon": horizon,
                "n": len(group),
                "contango_n": len(contango),
                "backwardation_n": len(backwardation),
                "contango_mean_excess_bp": contango.mean() * 10_000.0
                if len(contango)
                else np.nan,
                "backwardation_mean_excess_bp": backwardation.mean() * 10_000.0
                if len(backwardation)
                else np.nan,
                "backwardation_minus_contango_bp": (
                    backwardation.mean() - contango.mean()
                )
                * 10_000.0
                if len(contango) and len(backwardation)
                else np.nan,
                "contango_positive_fraction": float((contango > 0).mean())
                if len(contango)
                else np.nan,
                "backwardation_positive_fraction": float((backwardation > 0).mean())
                if len(backwardation)
                else np.nan,
            }
        )
    order = {label: idx for idx, label in enumerate(HORIZONS)}
    out = pd.DataFrame(rows)
    out["horizon_order"] = out["horizon"].map(order)
    return out.sort_values(["root", "target_months", "horizon_order"]).drop(
        columns="horizon_order"
    )


def build_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = load_signal_panel()
    horizons = load_horizon_panel()
    signal_cols = [
        "root",
        "target_months",
        "ts",
        "funding_rate",
        "funding_pct_ann",
        "funding_z_126d",
        "funding_change_24h_pct_ann",
        "funding_z_change_24h",
        "abs_funding_z_126d",
        "curve_state",
    ]
    data = horizons.merge(
        signals[signal_cols],
        on=["root", "target_months", "ts"],
        how="inner",
    )
    data = data.dropna(subset=["excess_after_path_funding_log"])
    rows = []
    decile_frames = []
    for (root, target, horizon), group in data.groupby(
        ["root", "target_months", "horizon"], sort=True
    ):
        horizon_bars = int(group["horizon_bars"].iloc[0])
        print(f"Testing {root} {target}M {horizon}", flush=True)
        for signal in SIGNALS:
            row, deciles = summarize_signal(
                root=root,
                target_months=target,
                horizon=horizon,
                horizon_bars=horizon_bars,
                signal=signal,
                data=group,
            )
            rows.append(row)
            if not deciles.empty:
                decile_frames.append(deciles)

    signal_summary = pd.DataFrame(rows)
    decile_summary = (
        pd.concat(decile_frames, ignore_index=True) if decile_frames else pd.DataFrame()
    )
    state_summary = summarize_state(data)
    return data, signal_summary, decile_summary, state_summary


def plot_signal_heatmaps(summary: pd.DataFrame, target_months: int, signal: str) -> None:
    data = summary[
        (summary["target_months"] == target_months) & (summary["signal"] == signal)
    ]
    if data.empty:
        return
    metrics = [
        (
            "high_minus_low_excess_bp",
            f"{signal}: high-minus-low decile excess return, target {target_months}M",
            f"{signal}_high_minus_low_target{target_months}m.png",
        ),
        (
            "hac_t",
            f"{signal}: HAC t-stat of standardized beta, target {target_months}M",
            f"{signal}_hac_t_target{target_months}m.png",
        ),
    ]
    for metric, title, filename in metrics:
        matrix = data.pivot(index="root", columns="horizon", values=metric).reindex(
            index=ROOTS,
            columns=HORIZONS,
        )
        values = matrix.to_numpy(dtype=float)
        finite = np.isfinite(values)
        vmax = np.nanpercentile(np.abs(values[finite]), 95) if finite.any() else 1.0
        vmax = max(vmax, 1e-9)
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(HORIZONS)), labels=HORIZONS)
        ax.set_yticks(np.arange(len(ROOTS)), labels=ROOTS)
        for i, root in enumerate(ROOTS):
            for j, horizon in enumerate(HORIZONS):
                value = matrix.loc[root, horizon]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax)
        fig.savefig(OUTPUT_DIR / filename, dpi=160)
        plt.close(fig)


def plot_state_heatmap(state_summary: pd.DataFrame, target_months: int) -> None:
    data = state_summary[state_summary["target_months"] == target_months]
    if data.empty:
        return
    matrix = data.pivot(
        index="root",
        columns="horizon",
        values="backwardation_minus_contango_bp",
    ).reindex(index=ROOTS, columns=HORIZONS)
    values = matrix.to_numpy(dtype=float)
    finite = np.isfinite(values)
    vmax = np.nanpercentile(np.abs(values[finite]), 95) if finite.any() else 1.0
    vmax = max(vmax, 1e-9)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title(f"Backwardation minus contango mean excess return, target {target_months}M")
    ax.set_xticks(np.arange(len(HORIZONS)), labels=HORIZONS)
    ax.set_yticks(np.arange(len(ROOTS)), labels=ROOTS)
    for i, root in enumerate(ROOTS):
        for j, horizon in enumerate(HORIZONS):
            value = matrix.loc[root, horizon]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="bp")
    fig.savefig(
        OUTPUT_DIR / f"state_backwardation_minus_contango_target{target_months}m.png",
        dpi=160,
    )
    plt.close(fig)


def plot_decile_profile(
    deciles: pd.DataFrame, target_months: int, horizon: str, signal: str
) -> None:
    data = deciles[
        (deciles["target_months"] == target_months)
        & (deciles["horizon"] == horizon)
        & (deciles["signal"] == signal)
    ]
    if data.empty:
        return
    fig, axes = plt.subplots(1, len(ROOTS), figsize=(17, 3.8), sharey=True, constrained_layout=True)
    for ax, root in zip(axes, ROOTS, strict=True):
        root_data = data[data["root"] == root].sort_values("decile")
        if root_data.empty:
            continue
        ax.bar(root_data["decile"], root_data["mean_excess_bp"], color="#2f7d8c", alpha=0.85)
        ax.axhline(0.0, color="#333333", lw=0.8)
        ax.set_title(root)
        ax.set_xlabel("decile")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("mean excess return bp")
    fig.suptitle(f"{signal} deciles: {horizon} excess return, target {target_months}M")
    fig.savefig(OUTPUT_DIR / f"{signal}_deciles_target{target_months}m_{horizon}.png", dpi=160)
    plt.close(fig)


def write_report(
    *,
    signal_summary: pd.DataFrame,
    state_summary: pd.DataFrame,
) -> None:
    top_abs_t = (
        signal_summary[signal_summary["n"] >= MIN_OBSERVATIONS]
        .assign(abs_hac_t=lambda df: df["hac_t"].abs())
        .sort_values("abs_hac_t", ascending=False)
        .head(20)
    )
    funding_z_3m_1d = signal_summary[
        (signal_summary["target_months"] == REPORT_TARGET_MONTHS)
        & (signal_summary["horizon"] == "1d")
        & (signal_summary["signal"] == "funding_z_126d")
    ]
    state_3m_1d = state_summary[
        (state_summary["target_months"] == REPORT_TARGET_MONTHS)
        & (state_summary["horizon"] == "1d")
    ]
    cols = [
        "root",
        "target_months",
        "horizon",
        "signal",
        "n",
        "beta_bp_per_1sd_signal",
        "hac_t",
        "pearson_corr",
        "spearman_corr",
        "low_decile_excess_bp",
        "high_decile_excess_bp",
        "high_minus_low_excess_bp",
        "high_positive_fraction",
        "low_positive_fraction",
    ]
    lines = [
        "# HYP-0038 Metals Funding Predictiveness",
        "",
        "## Question",
        "",
        "Does the current funding state forecast future realized return after realized "
        "funding paid?",
        "",
        "The dependent variable is:",
        "",
        "`forward_excess = forward_log_return - realized_path_funding_paid`",
        "",
        "## Signals",
        "",
        "- `funding_pct_ann`: current annualized curve-implied funding level.",
        "- `funding_z_126d`: current funding versus its own 126-day trailing history.",
        "- `funding_change_24h_pct_ann`: 24-hour change in annualized funding.",
        "- `funding_z_change_24h`: 24-hour change in standardized funding.",
        "- `abs_funding_z_126d`: absolute funding extremeness.",
        "",
        "## Method",
        "",
        "- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.",
        "- Tenors: `1M`, `3M`, `6M` front-to-deferred funding targets.",
        "- Horizons: `1h`, `4h`, `1d`, `3d`, `1w`, `1m`.",
        "- Regressions use standardized signals and HAC standard errors, capped at "
        f"{HAC_MAXLAGS_CAP} lags.",
        "- Decile spreads compare top 10% signal observations versus bottom 10%.",
        "",
        "## Top Absolute HAC t-Statistics",
        "",
        top_abs_t[cols].to_markdown(index=False, floatfmt=".4f")
        if not top_abs_t.empty
        else "No rows.",
        "",
        "## Funding z-score, 3M target, 1D horizon",
        "",
        funding_z_3m_1d[cols].to_markdown(index=False, floatfmt=".4f")
        if not funding_z_3m_1d.empty
        else "No rows.",
        "",
        "## Curve State, 3M target, 1D horizon",
        "",
        state_3m_1d.to_markdown(index=False, floatfmt=".4f")
        if not state_3m_1d.empty
        else "No rows.",
        "",
        "## Files",
        "",
        "- `signal_predictiveness_summary.csv`",
        "- `signal_decile_summary.csv`",
        "- `curve_state_summary.csv`",
        "- Heatmaps for 3M funding z-score, 24h funding change, and curve state.",
        "- Decile profile plots for 3M/1D funding z-score and 24h funding change.",
        "",
        "## Caveats",
        "",
        "- This is a broad multiple-testing screen; isolated t-stats should be treated "
        "as hypotheses, not evidence of tradable alpha.",
        "- Funding is still measured from the futures curve, not true spot/cash.",
        "- Overlapping horizons and regime clustering make the effective sample size "
        "much smaller than the raw row count.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data, signal_summary, decile_summary, state_summary = build_analysis()

    signal_summary.to_csv(OUTPUT_DIR / "signal_predictiveness_summary.csv", index=False)
    decile_summary.to_csv(OUTPUT_DIR / "signal_decile_summary.csv", index=False)
    state_summary.to_csv(OUTPUT_DIR / "curve_state_summary.csv", index=False)
    sample_cols = [
        "root",
        "target_months",
        "horizon",
        "ts",
        "funding_pct_ann",
        "funding_z_126d",
        "funding_change_24h_pct_ann",
        "funding_z_change_24h",
        "curve_state",
        "forward_log_return",
        "path_funding_paid_log",
        "excess_after_path_funding_log",
    ]
    data[sample_cols].to_parquet(OUTPUT_DIR / "predictiveness_aligned_panel.parquet", index=False)

    for signal in [
        "funding_z_126d",
        "funding_change_24h_pct_ann",
        "funding_z_change_24h",
        "abs_funding_z_126d",
    ]:
        plot_signal_heatmaps(signal_summary, target_months=3, signal=signal)
        plot_decile_profile(decile_summary, target_months=3, horizon="1d", signal=signal)
    plot_state_heatmap(state_summary, target_months=3)

    write_report(signal_summary=signal_summary, state_summary=state_summary)
    print(f"Aligned rows: {len(data):,}", flush=True)
    print(f"Signal summary rows: {len(signal_summary):,}", flush=True)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
