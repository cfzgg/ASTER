# `preprocess_data/dlpfc/` — benchmark inputs included in the archive

> These benchmark inputs are included in the single archive linked in the root README.
> This file documents the expected filenames and destination path.

These files are included in the root data archive and must be merged into this
directory. They are the benchmark-ready processed inputs — the reproducible starting point for
ASTER on DLPFC, so no upstream preprocessing is needed.

```text
preprocess_data/dlpfc/
├── 151507_preprocessed.h5ad   ~70 MB     one per slice, 12 slices
├── 151507_truth.txt           ~100 KB    manual layer annotation, one per slice
├── 151508_preprocessed.h5ad
├── 151508_truth.txt
│   ... 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675 ...
├── 151676_preprocessed.h5ad
└── 151676_truth.txt
```

24 files, ~867 MB total. Slice IDs: `151507 151508 151509 151510 151669 151670 151671
151672 151673 151674 151675 151676`.

Used by `notebooks/DLPFC_ASTER.ipynb` and `scripts/reproduce_dlpfc.sh`. The `_truth.txt`
files are only needed for the ARI evaluation (`--skip-mclust` omits it).

Verify your download with:

```bash
python scripts/check_data.py dlpfc
```
