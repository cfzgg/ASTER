#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m repro_st_aster.aster_ntd.run_dlpfc \
  --data-dir "$ROOT_DIR/preprocess_data/dlpfc" \
  --out-dir "$ROOT_DIR/results/dlpfc" \
  "$@"
