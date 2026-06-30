from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path("/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/raw")
OUTPUT_DIR = REPO_ROOT / "experiments" / "HYP-0036-metals-hourly-funding"

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
TARGET_MONTHS = [1, 3, 6]
MIN_DAILY_VOLUME = 10.0
MAX_ANCHOR_MONTHS_OUT = 4
MAX_CONTRACT_MONTHS_OUT = 120
MIN_HOURLY_VOLUME = 1.0
ROLLING_Z_DAYS = 126

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


def load_hourly_and_daily(root: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    path = RAW_DIR / f"{root}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    lazy = (
        pl.scan_parquet(path)
        .filter(~pl.col("symbol").str.contains("-"))
        .filter((pl.col("close") > 0) & (pl.col("volume") > 0))
    )
    daily = (
        lazy.with_columns(pl.col("ts_event").dt.date().alias("date"))
        .group_by(["date", "symbol"])
        .agg(
            [
                pl.col("close").sort_by("ts_event").last().alias("daily_last_close"),
                pl.col("volume").sum().alias("daily_volume"),
                pl.col("ts_event").max().alias("daily_last_ts"),
            ]
        )
        .collect()
        .to_pandas()
    )
    daily["date"] = pd.to_datetime(daily["date"], utc=True)
    daily["daily_last_ts"] = pd.to_datetime(daily["daily_last_ts"], utc=True)
    daily["root"] = root
    daily["months_out"] = [
        contract_months_out(symbol, date_value)
        for symbol, date_value in zip(daily["symbol"], daily["date"], strict=True)
    ]
    daily = daily.dropna(subset=["months_out", "daily_last_close", "daily_volume"])
    daily = daily[(daily["daily_last_close"] > 0) & (daily["daily_volume"] >= MIN_DAILY_VOLUME)]

    hourly = (
        lazy.with_columns(
            [
                pl.col("ts_event").dt.truncate("1h").alias("ts"),
                pl.col("ts_event").dt.date().alias("date"),
            ]
        )
        .group_by(["date", "ts", "symbol"])
        .agg(
            [
                pl.col("close").sort_by("ts_event").last().alias("close"),
                pl.col("volume").sum().alias("hourly_volume"),
                pl.col("ts_event").max().alias("last_ts"),
            ]
        )
        .collect()
        .to_pandas()
    )
    hourly["date"] = pd.to_datetime(hourly["date"], utc=True)
    hourly["ts"] = pd.to_datetime(hourly["ts"], utc=True)
    hourly["last_ts"] = pd.to_datetime(hourly["last_ts"], utc=True)
    hourly = hourly[(hourly["close"] > 0) & (hourly["hourly_volume"] >= MIN_HOURLY_VOLUME)]

    inventory = {
        "root": root,
        "daily_rows": len(daily),
        "hourly_rows": len(hourly),
        "first_ts": hourly["ts"].min(),
        "last_ts": hourly["ts"].max(),
        "contracts": daily["symbol"].nunique(),
        "first_date": daily["date"].min(),
        "last_date": daily["date"].max(),
    }
    return daily.sort_values(["date", "symbol"]), hourly.sort_values(
        ["date", "symbol", "ts"]
    ), inventory


def choose_daily_pairs(daily: pd.DataFrame, root: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in daily.groupby("date", sort=True):
        day = group[group["daily_volume"] >= MIN_DAILY_VOLUME].copy()
        early = day[day["months_out"] <= MAX_ANCHOR_MONTHS_OUT]
        if early.empty:
            continue
        anchor = early.sort_values(["daily_volume", "months_out"], ascending=[False, True]).iloc[
            0
        ]
        for target_months in TARGET_MONTHS:
            far_candidates = day[
                day["months_out"] - float(anchor["months_out"]) >= float(target_months)
            ]
            if far_candidates.empty:
                continue
            far = far_candidates.sort_values(
                ["months_out", "daily_volume"], ascending=[True, False]
            ).iloc[0]
            rows.append(
                {
                    "root": root,
                    "date": date,
                    "target_months": target_months,
                    "anchor_symbol": str(anchor["symbol"]),
                    "far_symbol": str(far["symbol"]),
                    "anchor_months_out": float(anchor["months_out"]),
                    "far_months_out": float(far["months_out"]),
                    "tenor_months": float(far["months_out"] - anchor["months_out"]),
                    "anchor_daily_volume": float(anchor["daily_volume"]),
                    "far_daily_volume": float(far["daily_volume"]),
                    "anchor_daily_close": float(anchor["daily_last_close"]),
                    "far_daily_close": float(far["daily_last_close"]),
                    "anchor_daily_last_ts": anchor["daily_last_ts"],
                    "far_daily_last_ts": far["daily_last_ts"],
                }
            )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs["row_id"] = np.arange(len(pairs), dtype=np.int64)
    return pairs


def build_hourly_funding_for_root(
    root: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    print(f"Building hourly funding for {root}", flush=True)
    daily, hourly, inventory = load_hourly_and_daily(root)
    pairs = choose_daily_pairs(daily, root)
    if pairs.empty:
        return pd.DataFrame(), pairs, inventory

    pair_cols = [
        "row_id",
        "root",
        "date",
        "target_months",
        "anchor_symbol",
        "far_symbol",
        "anchor_months_out",
        "far_months_out",
        "tenor_months",
        "anchor_daily_volume",
        "far_daily_volume",
    ]
    anchor = pairs[pair_cols].merge(
        hourly.rename(
            columns={
                "symbol": "anchor_symbol",
                "close": "anchor_price",
                "hourly_volume": "anchor_hourly_volume",
                "last_ts": "anchor_last_ts",
            }
        ),
        on=["date", "anchor_symbol"],
        how="inner",
    )
    far = pairs[["row_id", "date", "far_symbol"]].merge(
        hourly.rename(
            columns={
                "symbol": "far_symbol",
                "close": "far_price",
                "hourly_volume": "far_hourly_volume",
                "last_ts": "far_last_ts",
            }
        ),
        on=["date", "far_symbol"],
        how="inner",
    )
    funding = anchor.merge(
        far[
            [
                "row_id",
                "date",
                "ts",
                "far_symbol",
                "far_price",
                "far_hourly_volume",
                "far_last_ts",
            ]
        ],
        on=["row_id", "date", "ts", "far_symbol"],
        how="inner",
    )
    if funding.empty:
        return funding, pairs, inventory

    years = funding["tenor_months"].astype(float) / 12.0
    log_basis = np.log(funding["far_price"].astype(float) / funding["anchor_price"].astype(float))
    simple_basis = funding["far_price"].astype(float) / funding["anchor_price"].astype(float) - 1.0
    funding["tenor_years"] = years
    funding["log_basis"] = log_basis
    funding["simple_basis"] = simple_basis
    funding["funding_rate"] = log_basis / years
    funding["funding_pct_ann"] = funding["funding_rate"] * 100.0
    funding["funding_simple_pct_ann"] = simple_basis / years * 100.0
    funding["cash_minus_far_spread"] = funding["anchor_price"] - funding["far_price"]
    funding["curve_state"] = np.where(
        funding["funding_rate"].gt(0.0),
        "contango",
        np.where(funding["funding_rate"].lt(0.0), "backwardation", "flat"),
    )
    funding["long_carry_direction"] = np.where(
        funding["funding_rate"].gt(0.0),
        "long_pays",
        np.where(funding["funding_rate"].lt(0.0), "long_earns", "neutral"),
    )
    funding["common_hourly_volume"] = np.minimum(
        funding["anchor_hourly_volume"], funding["far_hourly_volume"]
    )
    return funding.sort_values(["root", "target_months", "ts"]), pairs, inventory


def add_daily_zscores(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = (
        hourly.groupby(["root", "target_months", "date"], as_index=False)
        .agg(
            median_funding_pct_ann=("funding_pct_ann", "median"),
            mean_funding_pct_ann=("funding_pct_ann", "mean"),
            obs=("funding_pct_ann", "size"),
            contango_fraction=("funding_rate", lambda values: float((values > 0).mean())),
            backwardation_fraction=("funding_rate", lambda values: float((values < 0).mean())),
            anchor_symbol=("anchor_symbol", lambda values: values.mode().iloc[0]),
            far_symbol=("far_symbol", lambda values: values.mode().iloc[0]),
            tenor_months=("tenor_months", "median"),
            common_hourly_volume=("common_hourly_volume", "median"),
        )
        .sort_values(["root", "target_months", "date"])
    )
    z_frames = []
    for (_root, _target), group in daily.groupby(["root", "target_months"], sort=False):
        data = group.copy()
        rolling = data["median_funding_pct_ann"].shift(1).rolling(
            ROLLING_Z_DAYS, min_periods=max(20, ROLLING_Z_DAYS // 4)
        )
        mean = rolling.mean()
        std = rolling.std(ddof=1)
        data["funding_z_126d"] = (data["median_funding_pct_ann"] - mean) / std.replace(
            0.0, np.nan
        )
        z_frames.append(data)
    daily = pd.concat(z_frames, ignore_index=True) if z_frames else daily
    hourly = hourly.merge(
        daily[["root", "target_months", "date", "funding_z_126d"]],
        on=["root", "target_months", "date"],
        how="left",
    )
    return hourly, daily


def summarize(hourly: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (root, target), group in hourly.groupby(["root", "target_months"], sort=True):
        daily_group = daily[(daily["root"] == root) & (daily["target_months"] == target)]
        rows.append(
            {
                "root": root,
                "target_months": target,
                "hourly_obs": len(group),
                "days": daily_group["date"].nunique(),
                "first_ts": group["ts"].min(),
                "last_ts": group["ts"].max(),
                "median_tenor_months": group["tenor_months"].median(),
                "median_funding_pct_ann": group["funding_pct_ann"].median(),
                "mean_funding_pct_ann": group["funding_pct_ann"].mean(),
                "p10_funding_pct_ann": group["funding_pct_ann"].quantile(0.10),
                "p90_funding_pct_ann": group["funding_pct_ann"].quantile(0.90),
                "contango_fraction": float((group["funding_rate"] > 0).mean()),
                "backwardation_fraction": float((group["funding_rate"] < 0).mean()),
                "median_abs_funding_z": daily_group["funding_z_126d"].abs().median(),
                "latest_ts": group["ts"].max(),
                "latest_funding_pct_ann": group.sort_values("ts")["funding_pct_ann"].iloc[-1],
                "latest_funding_z_126d": group.sort_values("ts")["funding_z_126d"].iloc[-1],
                "latest_anchor": group.sort_values("ts")["anchor_symbol"].iloc[-1],
                "latest_far": group.sort_values("ts")["far_symbol"].iloc[-1],
            }
        )
    return pd.DataFrame(rows)


def plot_daily_funding(daily: pd.DataFrame) -> None:
    for target in TARGET_MONTHS:
        data = daily[daily["target_months"] == target].copy()
        if data.empty:
            continue
        fig, axes = plt.subplots(
            len(ROOTS), 1, figsize=(14, 10), sharex=True, constrained_layout=True
        )
        for ax, root in zip(axes, ROOTS, strict=True):
            root_data = data[data["root"] == root].sort_values("date")
            if root_data.empty:
                continue
            ax.plot(
                root_data["date"],
                root_data["median_funding_pct_ann"],
                color=COLORS[root],
                lw=0.9,
                label="daily median hourly funding",
            )
            smooth = root_data["median_funding_pct_ann"].rolling(20, min_periods=5).median()
            ax.plot(root_data["date"], smooth, color="#1f2933", lw=1.1, label="20d median")
            ax.axhline(0, color="#333333", lw=0.8)
            ax.set_ylabel(f"{root}\n% ann.")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", fontsize=8)
        fig.suptitle(
            f"Hourly front-proxy funding, target >= {target}M deferred contract",
            fontsize=13,
        )
        axes[-1].set_xlabel("date")
        fig.savefig(OUTPUT_DIR / f"daily_median_hourly_funding_target{target}m.png", dpi=160)
        plt.close(fig)


def plot_latest_curve(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    pivot = summary.pivot(index="root", columns="target_months", values="latest_funding_pct_ann")
    pivot = pivot.reindex(index=ROOTS, columns=TARGET_MONTHS)
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    values = pivot.to_numpy(dtype=float)
    finite = np.isfinite(values)
    vmax = np.nanpercentile(np.abs(values[finite]), 90) if finite.any() else 1.0
    vmax = max(vmax, 1.0)
    image = ax.imshow(values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(TARGET_MONTHS)), labels=[f"{m}M" for m in TARGET_MONTHS])
    ax.set_yticks(np.arange(len(ROOTS)), labels=ROOTS)
    ax.set_title("Latest Hourly Funding Snapshot (% ann.)")
    for i, root in enumerate(ROOTS):
        for j, target in enumerate(TARGET_MONTHS):
            value = pivot.loc[root, target]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="% annualized")
    fig.savefig(OUTPUT_DIR / "latest_funding_snapshot_heatmap.png", dpi=160)
    plt.close(fig)


def write_report(
    *,
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    latest: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    summary_cols = [
        "root",
        "target_months",
        "hourly_obs",
        "days",
        "median_tenor_months",
        "median_funding_pct_ann",
        "p10_funding_pct_ann",
        "p90_funding_pct_ann",
        "contango_fraction",
        "backwardation_fraction",
        "latest_funding_pct_ann",
        "latest_funding_z_126d",
        "latest_anchor",
        "latest_far",
    ]
    lines = [
        "# HYP-0036 Metals Hourly Funding",
        "",
        "## Definition",
        "",
        "Funding is modeled as annualized curve-implied carry:",
        "",
        "`funding = ln(F_deferred / C_proxy) / T`",
        "",
        "where `C_proxy` is the selected near/prompt futures contract and `F_deferred` is the "
        "first liquid contract at least the target number of months beyond that anchor.",
        "",
        "- Positive funding: contango; long futures exposure pays carry.",
        "- Negative funding: backwardation; long futures exposure earns carry.",
        "- Units in the main CSV/report are percent annualized.",
        "",
        "## Method",
        "",
        "- Universe: `GC`, `SI`, `HG`, `PL`, `PA`.",
        "- Raw 1-minute outright futures are aggregated to hourly last marks by contract.",
        "- Calendar-spread symbols are excluded.",
        "- Anchor contract: highest daily-volume contract with `months_out <= 4`.",
        "- Deferred contract: first liquid contract at least `1M`, `3M`, or `6M` beyond anchor.",
        f"- Minimum daily contract volume: {MIN_DAILY_VOLUME:g}; minimum hourly contract volume: "
        f"{MIN_HOURLY_VOLUME:g}.",
        "- This is a front-futures proxy for spot/cash, not true cash funding.",
        "",
        "## Data Inventory",
        "",
        inventory.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Summary",
        "",
        summary[summary_cols].to_markdown(index=False, floatfmt=".4f")
        if not summary.empty
        else "No summary rows.",
        "",
        "## Latest Snapshot",
        "",
        latest[summary_cols].to_markdown(index=False, floatfmt=".4f")
        if not latest.empty
        else "No latest rows.",
        "",
        "## Pair Selection Coverage",
        "",
        pairs.groupby(["root", "target_months"], as_index=False)
        .agg(
            days=("date", "nunique"),
            median_tenor_months=("tenor_months", "median"),
            anchor_contracts=("anchor_symbol", "nunique"),
            far_contracts=("far_symbol", "nunique"),
        )
        .to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Caveats",
        "",
        "- Without true spot/cash prices, the estimate is front-to-deferred futures carry, not "
        "cash-to-futures carry.",
        "- Contract month timing is approximated from month codes; this is appropriate for a "
        "first research proxy but should be replaced with exact expiry/prompt calendars.",
        "- Daily pair selection uses same-day volume to identify liquid contracts. That is fine "
        "for curve measurement, but trading simulations should use prior-day or as-of-hour "
        "selection.",
        "- Palladium has sparse hourly coverage; PA funding estimates should receive stricter "
        "liquidity filters before being used for alpha research.",
        "",
        "## Files",
        "",
        "- `hourly_funding.parquet`",
        "- `hourly_funding.csv.gz`",
        "- `daily_funding.csv`",
        "- `funding_summary.csv`",
        "- `selected_pairs.csv`",
        "- `data_inventory.csv`",
        "- `daily_median_hourly_funding_target1m.png`",
        "- `daily_median_hourly_funding_target3m.png`",
        "- `daily_median_hourly_funding_target6m.png`",
        "- `latest_funding_snapshot_heatmap.png`",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    funding_frames = []
    pair_frames = []
    inventory_rows = []
    for root in ROOTS:
        funding, pairs, inventory = build_hourly_funding_for_root(root)
        inventory_rows.append(inventory)
        if not funding.empty:
            funding_frames.append(funding)
        if not pairs.empty:
            pair_frames.append(pairs)

    if not funding_frames:
        raise RuntimeError("No hourly funding rows were generated.")

    hourly = pd.concat(funding_frames, ignore_index=True).sort_values(
        ["root", "target_months", "ts"]
    )
    pairs = pd.concat(pair_frames, ignore_index=True).sort_values(
        ["root", "target_months", "date"]
    )
    inventory = pd.DataFrame(inventory_rows)

    hourly, daily = add_daily_zscores(hourly)
    summary = summarize(hourly, daily)
    latest = summary.loc[summary.groupby("root")["latest_ts"].idxmax()].copy()

    hourly.to_parquet(OUTPUT_DIR / "hourly_funding.parquet", index=False)
    hourly.to_csv(OUTPUT_DIR / "hourly_funding.csv.gz", index=False)
    daily.to_csv(OUTPUT_DIR / "daily_funding.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "funding_summary.csv", index=False)
    pairs.to_csv(OUTPUT_DIR / "selected_pairs.csv", index=False)
    inventory.to_csv(OUTPUT_DIR / "data_inventory.csv", index=False)

    plot_daily_funding(daily)
    plot_latest_curve(summary)
    write_report(inventory=inventory, summary=summary, latest=latest, pairs=pairs)
    print(f"Hourly funding rows: {len(hourly):,}", flush=True)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
