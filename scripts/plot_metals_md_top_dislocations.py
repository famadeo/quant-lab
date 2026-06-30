from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

ROOTS = ["GC", "SI", "HG", "PL", "PA"]
COLORS = {
    "GC": "#b8860b",
    "SI": "#6f7f8f",
    "HG": "#b15a2a",
    "PL": "#2f7d8c",
    "PA": "#7a4e9b",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPLORATION_DIR = REPO_ROOT / "notebooks" / "explorations"
CONTINUOUS_DIR = Path(
    "/home/famadeo/research/databento-asset-browser/data/metals_1m_10y/continuous"
)
ROLLING_EPISODES = (
    EXPLORATION_DIR
    / "assets"
    / "2026-06-25_metals_md_shock_return_paths_rolling30d"
    / "md_shock_episodes.csv"
)
MD_PATHS = {
    "5m": EXPLORATION_DIR
    / "assets"
    / "2026-06-25_metals_5m_trade_mahalanobis"
    / "trade_mahalanobis_5m.csv",
    "1h": EXPLORATION_DIR
    / "assets"
    / "2026-06-25_metals_trade_mahalanobis_hourly_daily"
    / "trade_mahalanobis_1h.csv",
}
WINDOWS = {
    "5m": {
        "pre": pd.Timedelta("6h"),
        "post": pd.Timedelta("12h"),
        "end_buffer": pd.Timedelta("1h"),
    },
    "1h": {
        "pre": pd.Timedelta("24h"),
        "post": pd.Timedelta("72h"),
        "end_buffer": pd.Timedelta("12h"),
    },
}
DEFAULT_OUTPUT_DIR = (
    EXPLORATION_DIR / "assets" / "2026-06-25_metals_md_top_dislocation_paths_rolling30d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top metal-complex MD dislocations with surrounding cumulative returns."
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--normal-quantile", type=float, default=0.75)
    parser.add_argument("--frequency", choices=["5m", "1h"], action="append")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_episodes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    episodes = pd.read_csv(path, parse_dates=["start_ts", "end_ts", "peak_ts"])
    for column in ["start_ts", "end_ts", "peak_ts"]:
        episodes[column] = pd.to_datetime(episodes[column], utc=True)
    return episodes


def load_md(freq: str) -> pd.DataFrame:
    path = MD_PATHS[freq]
    if not path.exists():
        raise FileNotFoundError(path)
    md = pd.read_csv(path, parse_dates=["ts"])
    md["ts"] = pd.to_datetime(md["ts"], utc=True)
    return md.set_index("ts").sort_index()


def load_1m_log_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = []
    for root in ROOTS:
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
        parts.append(frame.set_index("ts")["cont_logprice"].sort_index().rename(root))
    prices = pd.concat(parts, axis=1).sort_index().ffill()
    return prices.dropna(how="all")


def cumulative_from_start(prices: pd.DataFrame, start_ts: pd.Timestamp) -> pd.DataFrame:
    prices = prices.sort_index().ffill()
    base_pos = prices.index.searchsorted(start_ts, side="right") - 1
    base_pos = max(base_pos, 0)
    base = prices.iloc[base_pos]
    return prices.subtract(base, axis=1)


def format_timestamp(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M UTC")


def safe_name(freq: str, rank: int, start_ts: pd.Timestamp, peak_md: float) -> str:
    stamp = start_ts.strftime("%Y%m%d_%H%M")
    peak = f"{peak_md:.2f}".replace(".", "p")
    return f"{freq}_top_{rank:02d}_{stamp}_peak_md_{peak}.png"


def plot_event(
    *,
    freq: str,
    rank: int,
    event: pd.Series,
    md: pd.DataFrame,
    output_dir: Path,
) -> Path:
    start_ts = event["start_ts"]
    end_ts = event["end_ts"]
    peak_ts = event["peak_ts"]
    window = WINDOWS[freq]
    left = start_ts - window["pre"]
    right = max(start_ts + window["post"], end_ts + window["end_buffer"])

    md_window = md.loc[left:right].copy()
    prices = load_1m_log_prices(left, right)
    cum_bp = cumulative_from_start(prices, start_ts) * 10_000

    fig, (ax_md, ax_ret) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.35]},
        constrained_layout=True,
    )

    ax_md.plot(
        md_window.index,
        md_window["md_signed_flow_ewma"],
        color="#1f2933",
        lw=1.5,
        label="signed-flow MD",
    )
    ax_md.axhline(
        event["shock_threshold"],
        color="#c43d3d",
        lw=1.1,
        ls="--",
        label=f"shock q99 {event['shock_threshold']:.2f}",
    )
    ax_md.axhline(
        event["normal_threshold"],
        color="#287d4f",
        lw=1.1,
        ls="--",
        label=f"normal q75 {event['normal_threshold']:.2f}",
    )
    ax_md.axvspan(start_ts, end_ts, color="#f0b429", alpha=0.18, label="shock-to-normal")
    ax_md.axvline(start_ts, color="#c43d3d", lw=1.0)
    ax_md.axvline(peak_ts, color="#7c3aed", lw=1.0, ls=":")
    ax_md.axvline(end_ts, color="#287d4f", lw=1.0)
    ax_md.set_ylabel("Mahalanobis distance")
    ax_md.legend(loc="upper left", ncols=3, fontsize=8)

    title = (
        f"{freq} top {rank}: start {format_timestamp(start_ts)} | "
        f"peak MD {event['peak_md']:.2f} | "
        f"dominant {event['dominant_signed_root']} "
        f"({event['dominant_signed_share']:+.1%}) | "
        f"normal in {event['duration_minutes']:.0f} min"
    )
    ax_md.set_title(title, fontsize=12)

    for root in ROOTS:
        ax_ret.plot(cum_bp.index, cum_bp[root], lw=1.2, color=COLORS[root], label=root)
    ax_ret.axhline(0, color="#333333", lw=0.8)
    ax_ret.axvspan(start_ts, end_ts, color="#f0b429", alpha=0.18)
    ax_ret.axvline(start_ts, color="#c43d3d", lw=1.0)
    ax_ret.axvline(peak_ts, color="#7c3aed", lw=1.0, ls=":")
    ax_ret.axvline(end_ts, color="#287d4f", lw=1.0)
    ax_ret.set_ylabel("cum log return from shock start (bp)")
    ax_ret.set_xlabel("UTC time")
    ax_ret.legend(loc="upper left", ncols=len(ROOTS), fontsize=8)

    ax_ret.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    for ax in (ax_md, ax_ret):
        ax.grid(True, alpha=0.25)
        ax.margins(x=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / safe_name(freq, rank, start_ts, event["peak_md"])
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_index(output_dir: Path, rows: list[dict]) -> None:
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "top_dislocations_index.csv", index=False)

    cards = []
    for row in rows:
        rel_path = Path(row["plot"]).relative_to(output_dir)
        cards.append(
            "\n".join(
                [
                    "<section>",
                    f"<h2>{row['frequency']} top {row['rank']}: {row['start_ts']}</h2>",
                    (
                        f"<p>peak MD {row['peak_md']:.2f}; "
                        f"dominant {row['dominant_signed_root']} "
                        f"({row['dominant_signed_share']:+.1%}); "
                        f"duration {row['duration_minutes']:.0f} min</p>"
                    ),
                    f'<img src="{rel_path.as_posix()}" alt="{row["frequency"]} top {row["rank"]}">',
                    "</section>",
                ]
            )
        )
    html = "\n".join(
        [
            "<!doctype html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8">',
            "<title>Metals MD Top Dislocations</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;margin:24px;background:#f7f7f4;color:#1f2933}",
            "section{margin:0 0 32px 0;padding-bottom:24px;border-bottom:1px solid #d8d8d2}",
            "img{max-width:100%;height:auto;border:1px solid #d8d8d2;background:white}",
            "h1,h2{font-weight:600}",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Metals MD Top Dislocation Return Paths</h1>",
            *cards,
            "</body>",
            "</html>",
        ]
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    frequencies = args.frequency or ["1h", "5m"]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(ROLLING_EPISODES)
    index_rows = []
    for freq in frequencies:
        md = load_md(freq)
        selected = (
            episodes[
                (episodes["frequency"] == freq)
                & np.isclose(episodes["normal_quantile"], args.normal_quantile)
            ]
            .sort_values(["peak_md", "start_md"], ascending=False)
            .head(args.top_n)
            .reset_index(drop=True)
        )
        freq_dir = output_dir / freq
        for rank, (_, event) in enumerate(selected.iterrows(), start=1):
            path = plot_event(freq=freq, rank=rank, event=event, md=md, output_dir=freq_dir)
            index_rows.append(
                {
                    "frequency": freq,
                    "rank": rank,
                    "start_ts": event["start_ts"].isoformat(),
                    "end_ts": event["end_ts"].isoformat(),
                    "peak_ts": event["peak_ts"].isoformat(),
                    "start_md": event["start_md"],
                    "peak_md": event["peak_md"],
                    "end_md": event["end_md"],
                    "shock_threshold": event["shock_threshold"],
                    "normal_threshold": event["normal_threshold"],
                    "duration_minutes": event["duration_minutes"],
                    "dominant_signed_root": event["dominant_signed_root"],
                    "dominant_signed_share": event["dominant_signed_share"],
                    "plot": str(path),
                }
            )
    write_index(output_dir, index_rows)
    print(f"Wrote {len(index_rows)} plots and index to {output_dir}")


if __name__ == "__main__":
    main()
