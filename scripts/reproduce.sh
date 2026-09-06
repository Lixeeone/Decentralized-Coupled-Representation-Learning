#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python -m unittest discover -s tests -v
python -m dcrl run --config configs/quickstart.yaml
python -m dcrl graph --config configs/graph_circle.yaml
python -m dcrl lyapunov --output results/lyapunov
python -m dcrl baselines --config configs/baselines.yaml
