#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

EXPECTED_ARG_COUNT = 3


def main() -> int:
    if len(sys.argv) != EXPECTED_ARG_COUNT:
        print('usage: scripts/new_experiment.py HYP-0001 "Short title"', file=sys.stderr)
        return 2

    experiment_id = sys.argv[1]
    title = sys.argv[2]
    if not experiment_id.startswith("HYP-"):
        print("experiment id must look like HYP-0001", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[1]
    target = root / "experiments" / experiment_id
    if target.exists():
        print(f"{target} already exists", file=sys.stderr)
        return 1

    shutil.copytree(root / "experiments" / "templates", target)
    hypothesis_path = target / "hypothesis.md"
    hypothesis = hypothesis_path.read_text(encoding="utf-8")
    hypothesis = hypothesis.replace("{{EXPERIMENT_ID}}", experiment_id).replace("{{TITLE}}", title)
    hypothesis_path.write_text(hypothesis, encoding="utf-8")

    config_path = target / "config.yaml"
    config = config_path.read_text(encoding="utf-8")
    config = config.replace("{{EXPERIMENT_ID}}", experiment_id).replace("{{TITLE}}", title)
    config_path.write_text(config, encoding="utf-8")

    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
