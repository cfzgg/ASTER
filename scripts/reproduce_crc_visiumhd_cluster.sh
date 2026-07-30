#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

# Fig. 4a: latent smoothed over K=30 neighbours, KMeans K=15, tab20 colours.
python -m repro_st_aster.aster_sc.cluster_visualize \
  --data-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/bcam_input" \
  --bcam-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/bcam_output" \
  --inr-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/inr_output" \
  --vis-dir "$ROOT_DIR/preprocess_data/crc_visiumhd/viz" \
  --k 15 \
  --smooth-k 30 \
  --kmeans-random-state 0 \
  --kmeans-n-init auto \
  --palette none \
  --point-size 0.7 \
  --point-alpha 0.9 \
  --marker s \
  --invert-yaxis \
  --extra-labels-name kmeans_labels_K15.npy \
  "$@"
