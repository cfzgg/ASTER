#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

python -m repro_st_aster.uni_bcam.uni_extract \
  --image-path "$ROOT_DIR/raw_data/bc_xenium/tissue_standardized_0p5um.jpg" \
  --out-dir "$ROOT_DIR/preprocess_data/bc_xenium/uni" \
  "$@"
