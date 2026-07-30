#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m repro_st_aster.uni_bcam.prepare_inputs_visiumhd \
  --mtx-dir "$ROOT_DIR/raw_data/crc_visiumhd/binned_outputs/square_008um/filtered_feature_bc_matrix" \
  --positions-parquet "$ROOT_DIR/raw_data/crc_visiumhd/binned_outputs/square_008um/spatial/tissue_positions.parquet" \
  --he-metadata "$ROOT_DIR/raw_data/crc_visiumhd/metadata.json" \
  --uni-feature-path "$ROOT_DIR/preprocess_data/crc_visiumhd/uni/superpixel_features.npy" \
  --out-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/bcam_input" \
  --top-svg 2500 \
  --superpixel-stride 14 \
  "$@"
