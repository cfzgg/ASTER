# `raw_data/bc_xenium/` — files included in the archive

> These files are included in the single archive linked in the root README. This file
> documents the expected filenames and destination path; no separate download is needed.

This directory is populated from the root data archive. After merging the extracted
package into the repository root, these exact filenames must exist:

```text
raw_data/bc_xenium/
├── cell_feature_matrix.h5          18.7 MB   Xenium counts, 167,780 cells x 313 genes
├── cell_coordinates.csv            11.6 MB   x_standardized / y_standardized (0.5 um/px)
└── tissue_standardized_0p5um.jpg   23.9 MB   standardized H&E, 9408 x 6944 px
```

All three are required by `notebooks/BC_Xenium_Breast_Cancer_Demo.ipynb`. The
`.jpg` is additionally the input to UNI-2 extraction if you run route A
(`scripts/prepare_bc_xenium_uni.sh --model-dir ...`).

Verify your download with:

```bash
python scripts/check_data.py bc_xenium
```
