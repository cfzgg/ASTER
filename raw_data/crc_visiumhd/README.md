# `raw_data/crc_visiumhd/` — files included in the archive

> These files are included in the single archive linked in the root README. This file
> documents the expected filenames and destination path; no separate download is needed.

This directory ships **empty**. The colorectal cancer workflow
(`notebooks/CRC_VisiumHD_Colorectal_Cancer_Demo.ipynb`,
`scripts/*_crc_visiumhd_*.sh`) expects the following layout.

## 1. Raw Visium HD bundle — included in the root archive

Unpack so that these paths exist:

```text
raw_data/crc_visiumhd/binned_outputs/square_008um/
├── filtered_feature_bc_matrix/
│   ├── barcodes.tsv.gz         (~2.7 MB)
│   ├── features.tsv.gz         (~150 KB)
│   └── matrix.mtx.gz           (~810 MB)
└── spatial/
    └── tissue_positions.parquet (~11 MB)
```

545,913 in-tissue 8 µm bins x 18,085 genes.

## 2. Standardized H&E — included in the root archive

```text
raw_data/crc_visiumhd/
├── metadata.json                  (~2 KB)   required
└── tissue_standardized_0p5um.jpg  (~84 MB)  only needed for UNI-2 route A
```

`metadata.json` carries the crop bounding box and scale factor that map
full-resolution bin coordinates into standardized H&E space (0.5 µm/px); the
preprocessing step cannot run without it. The image itself is only read when you
extract UNI-2 features yourself.

## 3. UNI-2 features — included in the root archive

Goes to a different directory:

```text
preprocess_data/crc_visiumhd/uni/superpixel_features.npy   (912, 1008, 1536) float32, ~5.6 GB
```

Skip this only if you hold your own UNI-2 weights and set `HAVE_UNI2_WEIGHTS = True`
in the notebook (or run `scripts/prepare_crc_visiumhd_uni.sh --model-dir ...`). UNI-2
weights are not redistributed here — request them from the UNI-2 authors.

Verify your download with:

```bash
python scripts/check_data.py crc_visiumhd
```
