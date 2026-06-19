from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import duckdb
import typer
from rich.console import Console
from rich.table import Table

from quantlab.data.equity_futures import prepare_continuous_equity_futures
from quantlab.data.equity_universe import (
    DatabentoEquityUniverseConfig,
    build_databento_top_market_cap_universe,
)
from quantlab.experiments import run_experiment
from quantlab.pairs import run_pairs_experiment

app = typer.Typer(help="Quant Lab research commands.")
console = Console()


@app.command()
def doctor(quiet: bool = False, strict: bool = False) -> None:
    """Check local tool availability."""
    checks = [
        ("python", sys.executable, True),
        ("git", shutil.which("git"), True),
        ("uv", shutil.which("uv"), True),
        ("docker", shutil.which("docker"), False),
        ("quarto", shutil.which("quarto"), False),
        ("code", shutil.which("code"), False),
    ]

    missing_required = [name for name, path, required in checks if required and path is None]
    if not quiet:
        table = Table(title="Quant Lab Doctor")
        table.add_column("Tool")
        table.add_column("Status")
        table.add_column("Path")
        for name, path, required in checks:
            status = "ok" if path else ("missing required" if required else "missing optional")
            table.add_row(name, status, path or "")
        console.print(table)

    if strict and missing_required:
        raise typer.Exit(code=1)


@app.command("run-experiment")
def run_experiment_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    tracking_uri: Annotated[str | None, typer.Option(help="Override MLflow tracking URI.")] = None,
) -> None:
    """Run a configured research experiment and log it to MLflow."""
    result = run_experiment(config_path, tracking_uri=tracking_uri)
    console.print(f"[green]completed[/green] {result.experiment_id}")
    console.print(f"results: {result.results_path}")
    console.print(f"equity: {result.equity_curve_path}")
    console.print(f"mlflow_run_id: {result.mlflow_run_id}")
    console.print(result.metrics.to_dict())


@app.command("run-pairs-experiment")
def run_pairs_experiment_command(
    config_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    tracking_uri: Annotated[str | None, typer.Option(help="Override MLflow tracking URI.")] = None,
) -> None:
    """Run a configured pairs strategy comparison experiment and log it to MLflow."""
    result = run_pairs_experiment(config_path, tracking_uri=tracking_uri)
    console.print(f"[green]completed[/green] {result.experiment_id}")
    console.print(f"results: {result.results_path}")
    console.print(f"pair_selection: {result.pair_selection_path}")
    console.print(f"pair_metrics: {result.pair_metrics_path}")
    console.print(f"portfolio_returns: {result.portfolio_returns_path}")
    console.print(f"pair_returns: {result.pair_returns_path}")
    console.print(f"mlflow_run_id: {result.mlflow_run_id}")
    console.print({method: metrics.to_dict() for method, metrics in result.method_metrics.items()})


@app.command("build-equity-universe")
def build_equity_universe_command(
    output_path: Annotated[
        Path,
        typer.Option(help="CSV path for the ranked top market-cap equity universe."),
    ] = Path("data/bronze/databento/top100_us_equities.csv"),
    top_n: Annotated[int, typer.Option(help="Number of equities to keep.")] = 100,
    as_of: Annotated[
        str | None,
        typer.Option(help="As-of date for daily close lookup, formatted as YYYY-MM-DD."),
    ] = None,
    dataset: Annotated[
        str,
        typer.Option(help="Databento equity price dataset used for daily closes."),
    ] = "EQUS.MINI",
    price_lookback_days: Annotated[
        int,
        typer.Option(help="Lookback window used to find the latest available daily close."),
    ] = 7,
    symbols_per_request: Annotated[
        int,
        typer.Option(help="Symbol chunk size for Databento historical requests."),
    ] = 1_000,
    max_candidates: Annotated[
        int | None,
        typer.Option(help="Optional cap for smoke tests before ranking by market cap."),
    ] = None,
    env_file: Annotated[
        Path | None,
        typer.Option(help="Optional .env file containing DATABENTO_API_KEY."),
    ] = None,
    allow_billable: Annotated[
        bool,
        typer.Option(help="Required to run live Databento reference and price requests."),
    ] = False,
) -> None:
    """Build a top-N US equity universe ranked by Databento-derived market cap."""
    if not allow_billable:
        console.print(
            "[yellow]Refusing live Databento requests without --allow-billable.[/yellow]\n"
            "This command uses the Reference API plus equity daily bars to compute "
            "market cap = close * shares_outstanding."
        )
        raise typer.Exit(code=2)

    as_of_date = date.fromisoformat(as_of) if as_of else None
    universe = build_databento_top_market_cap_universe(
        DatabentoEquityUniverseConfig(
            top_n=top_n,
            dataset=dataset,
            as_of=as_of_date,
            price_lookback_days=price_lookback_days,
            symbols_per_request=symbols_per_request,
            max_candidates=max_candidates,
        ),
        output_path=output_path,
        env_file=env_file,
    )
    console.print(f"[green]wrote[/green] {len(universe)} symbols to {output_path}")
    console.print(universe.head(20))


@app.command("prepare-equity-futures-1m")
def prepare_equity_futures_1m_command(
    raw_dir: Annotated[
        Path,
        typer.Option(help="Directory containing GLBX.MDP3 root-level ohlcv-1m parquet files."),
    ] = Path("/home/famadeo/research/databento-asset-browser/data/equity_futures_1m"),
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory for continuous per-root parquet files."),
    ] = Path("data/silver/equity_futures_1m_continuous"),
    roots: Annotated[
        str | None,
        typer.Option(help="Optional comma-separated roots. Defaults to every parquet in raw-dir."),
    ] = None,
) -> None:
    """Convert outright equity futures 1-minute bars into continuous root series."""
    root_list = [root.strip() for root in roots.split(",") if root.strip()] if roots else None
    summaries = prepare_continuous_equity_futures(raw_dir, output_dir, roots=root_list)
    console.print(f"[green]wrote[/green] {len(summaries)} continuous roots to {output_dir}")
    table = Table(title="Continuous Equity Futures 1m")
    for column in ["root", "rows", "contracts", "rolls", "start", "end"]:
        table.add_column(column)
    for summary in summaries:
        table.add_row(
            summary.root,
            str(summary.rows),
            str(summary.contracts),
            str(summary.rolls),
            summary.start,
            summary.end,
        )
    console.print(table)


@app.command("render-report")
def render_report(
    report_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Render a Quarto report if Quarto is installed."""
    if shutil.which("quarto") is None:
        console.print("[red]quarto is not installed or not on PATH[/red]")
        raise typer.Exit(code=1)
    subprocess.run(["quarto", "render", str(report_path)], check=True)


@app.command("query")
def query_data(sql: str) -> None:
    """Run an ad hoc DuckDB SQL query."""
    with duckdb.connect() as con:
        result = con.sql(sql)
        console.print(result.df())
