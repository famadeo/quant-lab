from __future__ import annotations

import argparse
from pathlib import Path

from quantlab.metals_flow.config import MetalsFlowConfig
from quantlab.metals_flow.runner import run_metals_flow_research


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run metals flow anomaly research framework.")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("experiments/HYP-0012-metals-flow-anomaly-framework/config.yaml"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MetalsFlowConfig.from_yaml(args.config)
    result = run_metals_flow_research(config)
    print(f"wrote {result.results_path}")


if __name__ == "__main__":
    main()
