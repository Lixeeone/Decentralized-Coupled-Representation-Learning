#!/usr/bin/env bash
# Run compact reference diagnostics for the anonymous DCRL implementation.
# Outputs are written to ./results/.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p results

python3 reference_diagnostics.py \
  --output_dir results \
  --seed 0
