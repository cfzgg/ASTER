# `preprocess_data/simulation/` — benchmark inputs included in the archive

> These benchmark inputs are included in the single archive linked in the root README.
> This file documents the expected filenames and destination path.

These files are included in the root data archive and must be merged into this
directory. They are the benchmark-ready processed inputs — the reproducible starting point
for ASTER on the embryo simulation, so no upstream preprocessing is needed.

```text
preprocess_data/simulation/
├── embryo_section1_processed.h5ad   128 MB
├── embryo_section2_processed.h5ad   380 MB
└── embryo_section3_processed.h5ad   567 MB
```

3 files, ~1.1 GB total.

Used by `notebooks/Simulation_ASTER.ipynb` and `scripts/reproduce_simulation.sh`. Each
`.h5ad` carries the down-sampling scenarios as layers (e.g. `sr_norm_0.3`), which the
`--scenarios` flag selects.

Verify your download with:

```bash
python scripts/check_data.py simulation
```
