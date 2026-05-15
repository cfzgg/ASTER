#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m repro_st_aster.aster_sc.reconstruct \
  --data-dir "$ROOT_DIR/preprocess_data/bc_xenium/bcam_input" \
  --out-dir "$ROOT_DIR/preprocess_data/bc_xenium/inr_output" \
  "$@"
