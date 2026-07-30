#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

# Fig. 5: KMeans K=17 on the unsmoothed latent, published gastric palette.
python -m repro_st_aster.aster_sc.cluster_visualize \
  --data-dir "$ROOT_DIR/preprocess_data/gastric_xenium/bcam_input" \
  --bcam-dir "$ROOT_DIR/preprocess_data/gastric_xenium/bcam_output" \
  --inr-dir "$ROOT_DIR/preprocess_data/gastric_xenium/inr_output" \
  --vis-dir "$ROOT_DIR/preprocess_data/gastric_xenium/viz" \
  --k 17 \
  --smooth-k 0 \
  --kmeans-random-state 42 \
  --kmeans-n-init 20 \
  --palette fig5_gastric \
  --point-size 0.5 \
  --point-alpha 0.85 \
  --invert-yaxis \
  "$@"
