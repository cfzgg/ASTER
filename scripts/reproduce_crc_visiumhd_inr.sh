#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

# 8000-epoch schedule; the published reconstruction is the epoch-4000 snapshot.
python -m repro_st_aster.aster_sc.reconstruct_hd \
  --data-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/bcam_input" \
  --out-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/inr_output" \
  --epochs 8000 \
  --depth 8 \
  --recon-epoch 4000 \
  "$@"
