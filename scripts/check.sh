#!/usr/bin/env bash
set -euo pipefail

uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv run quantlab run-experiment experiments/HYP-0000-smoke/config.yaml
