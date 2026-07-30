# `raw_data/gastric_xenium/` — files included in the archive

> These files are included in the single archive linked in the root README. This file
> documents the expected filenames and destination path; no separate download is needed.

This directory ships **empty**. The gastric cancer workflow
(`notebooks/Gastric_Xenium_Gastric_Cancer_Demo.ipynb`,
`scripts/*_gastric_xenium_*.sh`) expects the following layout.

## 1. Raw Xenium counts — included in the root archive

```text
raw_data/gastric_xenium/cell_feature_matrix.h5   (~14 MB)
```

BS06 tumour section: 696,314 cells x the 377-gene panel.

## 2. Standardized H&E and cell coordinates — included in the root archive

```text
raw_data/gastric_xenium/
├── cell_coordinates.csv           (~85 MB)   required
└── tissue_standardized_0p5um.jpg  (~227 MB)  only needed for UNI-2 route A
```

`cell_coordinates.csv` provides the `x_standardized` / `y_standardized` columns (the
0.5 µm/px H&E space) that the workflow uses throughout; the row order matches the h5.
The image itself is only read when you extract UNI-2 features yourself.

## 3. UNI-2 features — included in the root archive

Goes to a different directory:

```text
preprocess_data/gastric_xenium/uni/superpixel_features.npy   (1168, 2864, 1536) float32, ~20.5 GB
```

Skip this only if you hold your own UNI-2 weights and set `HAVE_UNI2_WEIGHTS = True`
in the notebook (or run `scripts/prepare_gastric_xenium_uni.sh --model-dir ...`).
UNI-2 weights are not redistributed here — request them from the UNI-2 authors.

Verify your download with:

```bash
python scripts/check_data.py gastric_xenium
```
