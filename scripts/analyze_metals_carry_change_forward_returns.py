from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from scipy import stats

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
MIN_OBSERVATIONS = 30

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS_DIR = Path(
    "/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/continuous"
)
CARRY_PATH = (
    REPO_ROOT
    / "notebooks"
    / "explorations"
    / "assets"
    / "2026-06-25_metals_5m_carry_timeseries"
    / "annualized_carry_5m_exact_observations.csv"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "notebooks"
    / "explorations"
    / "assets"
    / "2026-06-25_metals_carry_change_forward_return_predictiveness"
)

FREQ_SPECS = {
    "5m": {
        "rule": "5min",
        "max_gap": pd.Timedelta("15min"),
        "horizons": {
            "5m": 1,
            "15m": 3,
            "30m": 6,
            "1h": 12,
            "4h": 48,
            "1d": 288,
        },
    },
    "1h": {
        "rule": "1h",
        "max_gap": pd.Timedelta("3h"),
        "horizons": {
            "1h": 1,
            "4h": 4,
            "1d": 24,
            "3d": 72,
            "5d": 120,
        },
    },
    "1d": {
        "rule": "1D",
        "max_gap": pd.Timedelta("4D"),
        "horizons": {
            "1d": 1,
            "3d": 3,
            "5d": 5,
            "10d": 10,
            "20d": 20,
        },
    },
}


def load_carry() -> pd.DataFrame:
    if not CARRY_PATH.exists():
        raise FileNotFoundError(CARRY_PATH)
    carry = pd.read_csv(CARRY_PATH, parse_dates=["ts"])
    carry["ts"] = pd.to_datetime(carry["ts"], utc=True)
    return carry.sort_values(["root", "ts"])


def load_log_price(root: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    path = CONTINUOUS_DIR / f"{root}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = (
        pl.scan_parquet(path)
        .filter((pl.col("ts") >= start) & (pl.col("ts") <= end))
        .select("ts", "cont_logprice")
        .collect()
        .to_pandas()
    )
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return frame.set_index("ts")["cont_logprice"].sort_index().rename(root)


def hac_ols_stats(x: pd.Series, y: pd.Series, maxlags: int) -> dict[str, float]:
    data = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(data) < MIN_OBSERVATIONS or data["x"].std() == 0 or data["y"].std() == 0:
        return {"beta": np.nan, "t": np.nan, "p": np.nan, "intercept": np.nan}
    model = sm.OLS(data["y"], sm.add_constant(data["x"]))
    result = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return {
        "intercept": float(result.params["const"]),
        "beta": float(result.params["x"]),
        "t": float(result.tvalues["x"]),
        "p": float(result.pvalues["x"]),
    }


def summarize_relation(
    *,
    root: str,
    frequency: str,
    horizon_label: str,
    horizon_steps: int,
    carry_change: pd.Series,
    forward_return: pd.Series,
) -> dict[str, float | int | str]:
    data = pd.concat(
        [carry_change.rename("carry_change_pct_ann"), forward_return.rename("forward_return")],
        axis=1,
    ).dropna()
    data = data[np.isfinite(data["carry_change_pct_ann"]) & np.isfinite(data["forward_return"])]
    if len(data) < MIN_OBSERVATIONS:
        return {
            "root": root,
            "frequency": frequency,
            "horizon": horizon_label,
            "n": len(data),
        }

    pearson = data["carry_change_pct_ann"].corr(data["forward_return"], method="pearson")
    spearman = stats.spearmanr(data["carry_change_pct_ann"], data["forward_return"]).statistic
    ols = hac_ols_stats(
        data["carry_change_pct_ann"],
        data["forward_return"],
        maxlags=max(1, horizon_steps - 1),
    )

    low_cut = data["carry_change_pct_ann"].quantile(0.1)
    high_cut = data["carry_change_pct_ann"].quantile(0.9)
    low = data[data["carry_change_pct_ann"] <= low_cut]["forward_return"]
    high = data[data["carry_change_pct_ann"] >= high_cut]["forward_return"]
    middle = data[
        (data["carry_change_pct_ann"] > low_cut) & (data["carry_change_pct_ann"] < high_cut)
    ]["forward_return"]

    return {
        "root": root,
        "frequency": frequency,
        "horizon": horizon_label,
        "horizon_steps": horizon_steps,
        "n": len(data),
        "carry_change_mean": data["carry_change_pct_ann"].mean(),
        "carry_change_std": data["carry_change_pct_ann"].std(),
        "forward_return_mean_bp": data["forward_return"].mean() * 10_000,
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "beta_bp_per_1pct_carry_change": ols["beta"] * 10_000,
        "hac_t": ols["t"],
        "hac_p": ols["p"],
        "low_decile_forward_bp": low.mean() * 10_000,
        "middle_forward_bp": middle.mean() * 10_000,
        "high_decile_forward_bp": high.mean() * 10_000,
        "high_minus_low_bp": (high.mean() - low.mean()) * 10_000,
        "high_positive_fraction": (high > 0).mean(),
        "low_positive_fraction": (low > 0).mean(),
    }


def build_frequency_inputs(
    carry_root: pd.DataFrame, price: pd.Series, frequency: str
) -> tuple[pd.Series, pd.Series]:
    spec = FREQ_SPECS[frequency]
    carry_series = (
        carry_root.set_index("ts")["annualized_log_carry_pct"]
        .sort_index()
        .resample(spec["rule"])
        .median()
        .dropna()
    )
    gaps = carry_series.index.to_series().diff()
    carry_change = carry_series.diff().where(gaps <= spec["max_gap"])

    price_series = price.resample(spec["rule"]).last().ffill()
    return carry_change.dropna(), price_series


def plot_heatmaps(summary: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("hac_t", "HAC t-stat of beta"),
        ("beta_bp_per_1pct_carry_change", "Beta: bp return per +1 pct annualized carry change"),
        ("high_minus_low_bp", "Top-minus-bottom carry-change decile forward return, bp"),
    ]
    for metric, title in metrics:
        fig, axes = plt.subplots(
            1,
            len(FREQ_SPECS),
            figsize=(16, 4.8),
            constrained_layout=True,
        )
        for ax, frequency in zip(axes, FREQ_SPECS, strict=True):
            data = summary[summary["frequency"] == frequency]
            horizons = list(FREQ_SPECS[frequency]["horizons"])
            matrix = data.pivot(index="root", columns="horizon", values=metric).reindex(
                index=ROOTS, columns=horizons
            )
            values = matrix.to_numpy(dtype=float)
            finite = np.isfinite(values)
            vmax = np.nanpercentile(np.abs(values[finite]), 95) if finite.any() else 1.0
            vmax = max(vmax, 1e-9)
            image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(frequency)
            ax.set_xticks(np.arange(len(horizons)))
            ax.set_xticklabels(horizons)
            ax.set_yticks(np.arange(len(ROOTS)))
            ax.set_yticklabels(ROOTS)
            for i, root in enumerate(ROOTS):
                for j, horizon in enumerate(horizons):
                    value = matrix.loc[root, horizon]
                    if np.isfinite(value):
                        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(title, fontsize=13)
        fig.savefig(output_dir / f"{metric}_heatmap.png", dpi=160)
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    carry = load_carry()
    start = carry["ts"].min() - pd.Timedelta("2D")
    end = carry["ts"].max() + pd.Timedelta("30D")

    rows = []
    aligned_frames = []
    for root in ROOTS:
        print(f"Processing {root}", flush=True)
        carry_root = carry[carry["root"] == root].copy()
        price = load_log_price(root, start, end)
        for frequency, spec in FREQ_SPECS.items():
            carry_change, price_series = build_frequency_inputs(carry_root, price, frequency)
            for horizon_label, steps in spec["horizons"].items():
                forward_return = price_series.shift(-steps) - price_series
                aligned = pd.concat(
                    [
                        carry_change.rename("carry_change_pct_ann"),
                        forward_return.reindex(carry_change.index).rename("forward_return"),
                    ],
                    axis=1,
                ).dropna()
                aligned.insert(0, "root", root)
                aligned.insert(1, "frequency", frequency)
                aligned.insert(2, "horizon", horizon_label)
                aligned = aligned.reset_index(names="ts")
                aligned_frames.append(aligned)
                rows.append(
                    summarize_relation(
                        root=root,
                        frequency=frequency,
                        horizon_label=horizon_label,
                        horizon_steps=steps,
                        carry_change=carry_change,
                        forward_return=forward_return.reindex(carry_change.index),
                    )
                )

    summary = pd.DataFrame(rows)
    aligned = pd.concat(aligned_frames, ignore_index=True)

    summary.to_csv(OUTPUT_DIR / "carry_change_forward_return_summary.csv", index=False)
    aligned.to_parquet(OUTPUT_DIR / "carry_change_forward_return_aligned.parquet", index=False)
    plot_heatmaps(summary, OUTPUT_DIR)

    print(summary.round(5).to_string(index=False))
    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
