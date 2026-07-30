#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m repro_st_aster.aster_sc.fusion \
  --data-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/bcam_input" \
  --inr-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/inr_output" \
  --out-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/bcam_output" \
  --epochs 30 \
  --k-neighbors 8 \
  "$@"
