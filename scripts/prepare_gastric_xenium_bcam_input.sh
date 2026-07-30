#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m repro_st_aster.uni_bcam.bcam_prepare_inputs \
  --matrix-h5 "$ROOT_DIR/raw_data/gastric_xenium/cell_feature_matrix.h5" \
  --coord-csv "$ROOT_DIR/raw_data/gastric_xenium/cell_coordinates.csv" \
  --uni-feature-path "$ROOT_DIR/preprocess_data/gastric_xenium/uni/superpixel_features.npy" \
  --out-dir "$ROOT_DIR/preprocess_data/gastric_xenium/bcam_input" \
  --superpixel-stride 16 \
  "$@"
