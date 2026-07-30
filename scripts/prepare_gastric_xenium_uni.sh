#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

# Gastric slide is 40096x16352; chunk each tile row so the GPU is not exhausted.
python -m repro_st_aster.uni_bcam.uni_extract \
  --image-path "$ROOT_DIR/raw_data/gastric_xenium/tissue_standardized_0p5um.jpg" \
  --out-dir "$ROOT_DIR/preprocess_data/gastric_xenium/uni" \
  --superpixel-stride 16 \
  --tile-batch-size 32 \
  --no-pickle \
  "$@"
