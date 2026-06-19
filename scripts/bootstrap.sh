#!/usr/bin/env bash
set -euo pipefail

uv sync --all-extras --group dev
cp -n .env.example .env
uv run pre-commit install

echo "Quant Lab is ready."
echo "Run ./scripts/check.sh to verify the environment."
