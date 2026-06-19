from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
EXPERIMENTS_DIR = ROOT / "experiments"
PUBLIC_EXPERIMENTS_DIR = DOCS_DIR / "experiments"
PUBLIC_RESULTS_DIR = DOCS_DIR / "results"
PROPRIETARY_NOTICE = (
    "This page is part of Francisco Amadeo's proprietary quantitative research "
    "record. Public access is provided for review and documentation only; no "
    "license is granted to copy, reuse, redistribute, commercialize, or implement "
    "the research, strategy logic, or derived conclusions."
)


def main() -> None:
    registry = _load_yaml(EXPERIMENTS_DIR / "registry.yaml")
    experiment_entries = registry.get("experiments", [])
    if not isinstance(experiment_entries, list):
        raise ValueError("experiments/registry.yaml must contain an experiments list")

    PUBLIC_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [_build_experiment(entry) for entry in experiment_entries]
    for experiment in experiments:
        page = _render_experiment_page(experiment)
        (PUBLIC_EXPERIMENTS_DIR / f"{experiment['id']}.md").write_text(page, encoding="utf-8")

    (PUBLIC_EXPERIMENTS_DIR / "index.md").write_text(
        _render_experiment_index(experiments), encoding="utf-8"
    )
    (PUBLIC_RESULTS_DIR / "index.md").write_text(
        _render_results_index(experiments), encoding="utf-8"
    )
    print(f"wrote {len(experiments)} experiment wiki pages")


def _build_experiment(entry: dict[str, Any]) -> dict[str, Any]:
    experiment_id = str(entry["id"])
    experiment_dir = EXPERIMENTS_DIR / experiment_id
    config = (
        _load_yaml(experiment_dir / "config.yaml")
        if (experiment_dir / "config.yaml").exists()
        else {}
    )
    result_files = _load_result_files(experiment_dir)
    strict_selection = _load_json_if_exists(experiment_dir / "strict_selection.json")
    decision = _latest_decision(config, result_files, entry)
    return {
        "id": experiment_id,
        "title": str(config.get("title") or entry.get("title") or experiment_id),
        "conceptual_description": str(
            config.get("conceptual_description")
            or entry.get("conceptual_description")
            or "No conceptual description recorded."
        ),
        "status": str(decision.get("status") or entry.get("status") or "unknown"),
        "owner": str(entry.get("owner") or ""),
        "notes": str(decision.get("notes") or entry.get("notes") or ""),
        "config": config,
        "result_files": result_files,
        "strict_selection": strict_selection,
        "experiment_dir": experiment_dir,
    }


def _load_result_files(experiment_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(experiment_dir.glob("*results.json")):
        payload = _load_json_if_exists(path)
        if payload:
            label = "primary" if path.name == "results.json" else path.stem.replace("_", " ")
            results.append({"label": label, "path": path, "payload": payload})
    return results


def _latest_decision(
    config: dict[str, Any],
    result_files: list[dict[str, Any]],
    entry: dict[str, Any],
) -> dict[str, Any]:
    for result_file in reversed(result_files):
        decision = result_file["payload"].get("decision")
        if isinstance(decision, dict):
            return decision
    decision = config.get("decision")
    if isinstance(decision, dict):
        return decision
    return {"status": entry.get("status"), "notes": entry.get("notes")}


def _render_experiment_index(experiments: list[dict[str, Any]]) -> str:
    rows = [
        "| ID | Title | Concept | Status | Best Result | Notes |",
        "|---|---|---|---|---:|---|",
    ]
    for experiment in experiments:
        best = _best_result(experiment["result_files"])
        rows.append(
            "| "
            + " | ".join(
                [
                    f"[{experiment['id']}]({experiment['id']}.md)",
                    _escape_table(str(experiment["title"])),
                    _escape_table(_brief_description(str(experiment["conceptual_description"]))),
                    _escape_table(str(experiment["status"])),
                    _format_percent(best["total_return"]) if best else "",
                    _escape_table(str(experiment["notes"])),
                ]
            )
            + " |"
        )
    return "\n".join(
        [
            "# Experiment Index",
            "",
            "Generated from `experiments/registry.yaml` and local result artifacts.",
            "",
            PROPRIETARY_NOTICE,
            "",
            *rows,
            "",
        ]
    )


def _render_results_index(experiments: list[dict[str, Any]]) -> str:
    rows = [
        "| Experiment | Result File | Method | Total Return | Sharpe | Max Drawdown | Trades |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for experiment in experiments:
        for result_file in experiment["result_files"]:
            for method, metrics in _method_metrics(result_file["payload"]).items():
                rows.append(
                    "| "
                    + " | ".join(
                        [
                            f"[{experiment['id']}](../experiments/{experiment['id']}.md)",
                            _escape_table(str(result_file["label"])),
                            _display_method(method),
                            _format_percent(metrics.get("total_return")),
                            _format_number(metrics.get("sharpe_ratio")),
                            _format_percent(metrics.get("max_drawdown")),
                            _format_int(metrics.get("trades")),
                        ]
                    )
                    + " |"
                )
    return "\n".join(
        [
            "# Overall Results",
            "",
            PROPRIETARY_NOTICE,
            "",
            "This page aggregates publishable method-level metrics. It is not a promotion "
            "list; every result must still pass the promotion gates before it can be "
            "treated as evidence of edge.",
            "",
            *rows,
            "",
        ]
    )


def _render_experiment_page(experiment: dict[str, Any]) -> str:
    config = experiment["config"]
    lines = [
        f"# {experiment['id']}: {experiment['title']}",
        "",
        f"> {PROPRIETARY_NOTICE}",
        "",
        f"- Status: `{experiment['status']}`",
        f"- Owner: `{experiment['owner'] or 'unassigned'}`",
        f"- Decision notes: {_escape_inline(experiment['notes'])}",
        "",
        "## Hypothesis",
        "",
        str(config.get("hypothesis") or "No hypothesis recorded.").strip(),
        "",
        "## Conceptual Description",
        "",
        experiment["conceptual_description"].strip(),
        "",
        "## Experiment Design",
        "",
        *_render_design(config),
        "",
    ]
    if experiment["strict_selection"]:
        lines.extend(_render_strict_selection(experiment["strict_selection"]))
        lines.append("")
    if experiment["result_files"]:
        for result_file in experiment["result_files"]:
            lines.extend(_render_result_section(result_file["label"], result_file["payload"]))
            lines.append("")
    else:
        lines.extend(
            ["## Results", "", "No result artifact was available when the wiki was built.", ""]
        )

    lines.extend(
        [
            "## Publication Notes",
            "",
            "- Proprietary work by Francisco Amadeo. All rights reserved.",
            "- Public access does not grant permission to copy, reuse, redistribute, "
            "commercialize, or implement this research.",
            "- Local data roots and artifact paths are intentionally omitted.",
            "- Raw data, parquet outputs, MLflow state, and credentials are not published.",
            "- This page is a summary; the experiment folder remains the source of truth.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_design(config: dict[str, Any]) -> list[str]:
    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    strategy = config.get("strategy") if isinstance(config.get("strategy"), dict) else {}
    selection = config.get("selection") if isinstance(config.get("selection"), dict) else {}
    backtest = config.get("backtest") if isinstance(config.get("backtest"), dict) else {}
    roots = data.get("roots") if isinstance(data, dict) else []
    asset_classes = data.get("asset_classes") if isinstance(data, dict) else {}
    lines = [
        f"- Roots: {_format_list(roots)}",
        f"- Asset groups: {_format_asset_classes(asset_classes)}",
        f"- Pair scope: `{strategy.get('pair_scope', 'n/a')}`",
        f"- Lookback: `{strategy.get('lookback', 'n/a')}` bars",
        f"- Signal lag: `{strategy.get('signal_lag', 'n/a')}` bars",
        f"- Rebalance interval: `{strategy.get('rebalance_every_bars', 'n/a')}` bars",
        f"- Selection enabled: `{selection.get('enabled', 'n/a')}`",
        f"- Train fraction: `{selection.get('train_fraction', 'n/a')}`",
        f"- Fee bps: `{backtest.get('fee_bps', 'n/a')}`",
        f"- Slippage bps: `{backtest.get('slippage_bps', 'n/a')}`",
    ]
    return lines


def _render_strict_selection(selection: dict[str, Any]) -> list[str]:
    rows = [
        "| Pair | Selected | Reason | Observations | ADF p-value | Half-life bars |",
        "|---|---:|---|---:|---:|---:|",
        "| "
        + " | ".join(
            [
                _escape_table(str(selection.get("pair", ""))),
                str(selection.get("selected", "")),
                _escape_table(str(selection.get("reason", ""))),
                _format_int(selection.get("observations")),
                _format_number(selection.get("spread_adf_pvalue")),
                _format_number(selection.get("half_life_bars")),
            ]
        )
        + " |",
    ]
    return ["## Strict Selection", "", *rows]


def _render_result_section(label: str, payload: dict[str, Any]) -> list[str]:
    section_title = "Results" if label == "primary" else f"Results: {label.title()}"
    lines = [f"## {section_title}", ""]
    completed_at = payload.get("completed_at")
    if completed_at:
        lines.extend([f"- Completed at: `{completed_at}`"])
    if "candidate_pairs" in payload:
        lines.append(f"- Candidate pairs: `{payload.get('candidate_pairs')}`")
    if "selected_pairs" in payload:
        lines.append(f"- Selected pairs: `{payload.get('selected_pairs')}`")
    lines.append("")
    metrics = _method_metrics(payload)
    if metrics:
        lines.extend(_render_metrics_table(metrics))
    elif isinstance(payload.get("metrics"), dict):
        lines.extend(_render_single_metrics_table(payload["metrics"]))
    selection_reasons = payload.get("selection_reasons")
    if isinstance(selection_reasons, dict) and selection_reasons:
        lines.extend(["", "### Selection Reasons", "", *_render_key_value_table(selection_reasons)])
    top_pairs = payload.get("top_pairs_by_sharpe")
    if isinstance(top_pairs, list) and top_pairs:
        lines.extend(["", "### Top Pairs By Sharpe", "", *_render_top_pairs(top_pairs)])
    return lines


def _render_metrics_table(metrics: dict[str, dict[str, Any]]) -> list[str]:
    rows = [
        "| Method | Total Return | Sharpe | Max Drawdown | Active Fraction | Turnover | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in sorted(metrics.items()):
        rows.append(
            "| "
            + " | ".join(
                [
                    _display_method(method),
                    _format_percent(values.get("total_return")),
                    _format_number(values.get("sharpe_ratio")),
                    _format_percent(values.get("max_drawdown")),
                    _format_percent(values.get("active_fraction")),
                    _format_number(values.get("total_turnover")),
                    _format_int(values.get("trades")),
                ]
            )
            + " |"
        )
    return rows


def _render_single_metrics_table(metrics: dict[str, Any]) -> list[str]:
    return [
        "| Metric | Value |",
        "|---|---:|",
        *[
            f"| {_escape_table(str(key))} | {_format_metric_value(value)} |"
            for key, value in sorted(metrics.items())
        ],
    ]


def _render_key_value_table(values: dict[str, Any]) -> list[str]:
    rows = ["| Key | Value |", "|---|---:|"]
    for key, value in sorted(values.items()):
        rows.append(f"| {_escape_table(str(key))} | {_format_metric_value(value)} |")
    return rows


def _render_top_pairs(top_pairs: list[dict[str, Any]], limit: int = 10) -> list[str]:
    rows = [
        "| Pair | Method | Asset Class | Total Return | Sharpe | Trades |",
        "|---|---|---|---:|---:|---:|",
    ]
    rows.extend(
        (
            "| "
            + " | ".join(
                [
                    _escape_table(str(row.get("pair", ""))),
                    _display_method(str(row.get("method", ""))),
                    _escape_table(str(row.get("asset_class", ""))),
                    _format_percent(row.get("total_return")),
                    _format_number(row.get("sharpe_ratio")),
                    _format_int(row.get("trades")),
                ]
            )
            + " |"
        )
        for row in top_pairs[:limit]
    )
    return rows


def _method_metrics(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = payload.get("method_metrics")
    return metrics if isinstance(metrics, dict) else {}


def _best_result(result_files: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for result_file in result_files:
        for method, metrics in _method_metrics(result_file["payload"]).items():
            if isinstance(metrics, dict) and isinstance(metrics.get("total_return"), int | float):
                candidates.append(
                    {
                        "method": method,
                        "total_return": float(metrics["total_return"]),
                        "label": result_file["label"],
                    }
                )
    return max(candidates, key=lambda row: row["total_return"]) if candidates else None


def _format_asset_classes(asset_classes: Any) -> str:
    if not isinstance(asset_classes, dict) or not asset_classes:
        return "`n/a`"
    parts = []
    for name, roots in asset_classes.items():
        parts.append(f"{name} ({len(roots) if isinstance(roots, list) else 0})")
    return ", ".join(parts)


def _format_list(values: Any, limit: int = 30) -> str:
    if not isinstance(values, list):
        return "`n/a`"
    rendered = [str(value) for value in values[:limit]]
    suffix = f", +{len(values) - limit} more" if len(values) > limit else ""
    return "`" + ", ".join(rendered) + suffix + "`"


def _format_metric_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return _format_number(value)
    return _escape_table(str(value))


def _format_percent(value: Any) -> str:
    return "" if not isinstance(value, int | float) else f"{float(value):.2%}"


def _format_number(value: Any) -> str:
    return "" if not isinstance(value, int | float) else f"{float(value):.2f}"


def _format_int(value: Any) -> str:
    return "" if not isinstance(value, int | float) else f"{int(value):,}"


def _display_method(method: str) -> str:
    labels = {
        "zscore": "Z-score",
        "mahalanobis": "Mahalanobis",
        "zscore_mahalanobis": "Z-score + Mahalanobis filter",
    }
    return labels.get(method, _escape_table(method))


def _brief_description(value: str, limit: int = 180) -> str:
    summary = value.replace("\n", " ").strip()
    first_sentence = summary.split(". ", 1)[0].strip()
    if first_sentence:
        summary = first_sentence + "."
    if len(summary) <= limit:
        return summary
    return summary[: limit - 3].rstrip() + "..."


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _escape_inline(value: str) -> str:
    return value.replace("\n", " ").strip()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
