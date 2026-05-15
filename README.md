# repro-st-aster

Reproducible GPU-oriented repository for three workflows related to the ASTER paper family:

1. `simulation`: ASTER neural Tucker decomposition on embryo simulation benchmarks
2. `dlpfc`: ASTER neural Tucker decomposition on 12 DLPFC slices
3. `bc_xenium`: breast cancer Xenium demo covering UNI feature extraction, BCAM, Tucker-2 INR, and ASTER-SC style clustering

## Repository layout

```text
repro-st-aster/
├── src/repro_st_aster/
│   ├── aster_ntd/      # Tucker-3 ASTER code for simulation + DLPFC
│   ├── aster_sc/       # Tucker-2 INR + ASTER-SC breast cancer pipeline
│   ├── uni_bcam/       # UNI extraction + BCAM input/training code
│   └── common/         # shared runtime helpers
├── raw_data/
│   ├── bc_xenium/      # raw breast cancer Xenium inputs
│   ├── dlpfc/          # optional place for upstream raw downloads
│   └── simulation/     # optional place for upstream raw downloads
├── preprocess_data/
│   ├── bc_xenium/      # UNI output, BCAM input, INR output, BCAM output, viz
│   ├── dlpfc/          # preprocessed 12-slice inputs used by ASTER
│   └── simulation/     # preprocessed embryo benchmark inputs used by ASTER
├── notebooks/          # visualization-oriented notebooks
├── scripts/            # shell entrypoints for reproduction
├── results/            # generated outputs
└── docs/
```

## Working dependency baseline

The tested baseline in this workspace is Python `3.9.23`.
- `torch==2.1.0+cu121`
- `torchvision==0.16.0+cu121`
- `timm==1.0.22`
- `numpy==1.26.3`
- `pandas==2.3.3`
- `scanpy==1.10.3`
- `anndata==0.10.9`
- `scipy==1.13.1`
- `scikit-learn==1.6.1`
- `tqdm==4.67.1`
- `matplotlib==3.9.4`
- `Pillow==11.3.0`
- `scikit-misc==0.3.1`
- `tensorly==0.9.0`



## Installation

Recommended:

```bash
conda create -n repro_st_aster python=3.9 -y
conda activate repro_st_aster
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

Optional DLPFC ARI dependency:

```bash
python -m pip install rpy2==3.5.11
```

The DLPFC ARI step also requires a system R installation with the `mclust` package available.

## Data convention

### `raw_data`

- `raw_data/bc_xenium/`
  - `cell_feature_matrix.h5`
  - `cell_coordinates.csv`
  - `tissue_standardized_0p5um.jpg`
- `raw_data/dlpfc/`
  - reserved for optional upstream raw downloads
- `raw_data/simulation/`
  - reserved for optional upstream raw downloads

### `preprocess_data`

The data required under `preprocess_data/` can be downloaded from Google Drive:

https://drive.google.com/drive/folders/1OQudu7wDPlJwyDIstHzmjBuXqyxFT_xO?usp=sharing

After downloading, place the extracted folders/files under `preprocess_data/` following the structure below.

- `preprocess_data/bc_xenium/uni/`
  - UNI feature maps
- `preprocess_data/bc_xenium/bcam_input/`
  - normalized expression, per-cell UNI features, coordinates
- `preprocess_data/bc_xenium/inr_output/`
  - Tucker-2 INR outputs
- `preprocess_data/bc_xenium/bcam_output/`
  - BCAM fusion outputs
- `preprocess_data/bc_xenium/viz/`
  - clustering labels and visualization artifacts
- `preprocess_data/dlpfc/`
  - `*_preprocessed.h5ad`, `*_truth.txt`
- `preprocess_data/simulation/`
  - `*_processed.h5ad`

For DLPFC and simulation, this repository currently ships the benchmark-ready processed inputs. In other words, those files are the reproducible starting point for ASTER itself inside this repo.

## Reproduction commands

### 1. Breast cancer Xenium

Prepare UNI features:

```bash
bash scripts/prepare_bc_xenium_uni.sh --model-dir /path/to/uni2-h
```

Prepare BCAM inputs:

```bash
bash scripts/prepare_bc_xenium_bcam_input.sh
```

Run Tucker-2 INR:

```bash
bash scripts/reproduce_bc_xenium_inr.sh
```

Run BCAM fusion:

```bash
bash scripts/reproduce_bc_xenium_bcam.sh
```

Run clustering and summary visualization:

```bash
bash scripts/reproduce_bc_xenium_cluster.sh
```

Generate the notebook figures:

```bash
bash scripts/reproduce_bc_xenium_notebook_examples.sh
```

### 2. Simulation

```bash
bash scripts/reproduce_simulation.sh
```

### 3. DLPFC

With ARI:

```bash
bash scripts/reproduce_dlpfc.sh --r-home /path/to/R
```

Without `mclust`:

```bash
bash scripts/reproduce_dlpfc.sh --skip-mclust
```

## Notebooks

The notebooks are intended for result review and figure reproduction from `preprocess_data`, not for hiding core training logic.

- `notebooks/Simulation_ASTER.ipynb`
- `notebooks/DLPFC_ASTER.ipynb`
- `notebooks/BC_Xenium_Breast_Cancer_Demo.ipynb`

## UNI note

The repository includes precomputed UNI-derived features for the breast cancer demo so readers can reproduce downstream steps directly. UNI model weights are not redistributed here; users must request them separately if they want to rerun feature extraction from scratch.

## Smoke tests

Breast cancer:

```bash
bash scripts/reproduce_bc_xenium_inr.sh --epochs 1 --batch-size 128 --max-cells 512 --out-dir /tmp/bc_inr_smoke
bash scripts/reproduce_bc_xenium_bcam.sh --epochs 1 --batch-size 64 --max-cells 512 --inr-dir /tmp/bc_inr_smoke --out-dir /tmp/bc_bcam_smoke
bash scripts/reproduce_bc_xenium_cluster.sh --max-cells 2000 --inr-dir /tmp/bc_inr_smoke --bcam-dir /tmp/bc_bcam_smoke --vis-dir /tmp/bc_cluster_smoke
```

Simulation:

```bash
bash scripts/reproduce_simulation.sh --sections embryo_section1 --scenarios sr_norm_0.3 --max-epoch 2
```

DLPFC:

```bash
bash scripts/reproduce_dlpfc.sh --slice-ids 151507 --max-epoch 2 --skip-mclust
```
