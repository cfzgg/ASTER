#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m repro_st_aster.aster_sc.notebook_examples \
  --raw-dir "$ROOT_DIR/raw_data/bc_xenium" \
  --preprocess-dir "$ROOT_DIR/preprocess_data/bc_xenium" \
  --out-dir "$ROOT_DIR/results/BC_Xenium_notebook_examples" \
  "$@"
