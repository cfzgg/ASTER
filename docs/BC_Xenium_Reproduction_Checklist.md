# BC Xenium Reproduction Checklist

## Environment

Recommended:

```bash
conda create -n repro_st_aster python=3.9 -y
conda activate repro_st_aster
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt
```

If you insist on `single_cell_python`, first repair its `torch` / `torchvision` mismatch and install `timm`.

## Repository-local paths

Raw input:

- `raw_data/bc_xenium/cell_feature_matrix.h5`
- `raw_data/bc_xenium/cell_coordinates.csv`
- `raw_data/bc_xenium/tissue_standardized_0p5um.jpg`

Preprocessed outputs:

- `preprocess_data/bc_xenium/uni/`
- `preprocess_data/bc_xenium/bcam_input/`
- `preprocess_data/bc_xenium/inr_output/`
- `preprocess_data/bc_xenium/bcam_output/`
- `preprocess_data/bc_xenium/viz/`

## Reproduction entry points

### 1. UNI extraction

```bash
bash scripts/prepare_bc_xenium_uni.sh --model-dir /path/to/uni2-h
```

### 2. BCAM input preparation

```bash
bash scripts/prepare_bc_xenium_bcam_input.sh
```

### 3. INR reconstruction

```bash
bash scripts/reproduce_bc_xenium_inr.sh
```

### 4. BCAM fusion

```bash
bash scripts/reproduce_bc_xenium_bcam.sh
```

### 5. Clustering and visualization

```bash
bash scripts/reproduce_bc_xenium_cluster.sh
```

### 6. Notebook figure export

```bash
bash scripts/reproduce_bc_xenium_notebook_examples.sh
```

## Smoke tests

```bash
bash scripts/reproduce_bc_xenium_inr.sh --epochs 1 --batch-size 128 --max-cells 512 --out-dir /tmp/bc_inr_smoke
bash scripts/reproduce_bc_xenium_bcam.sh --epochs 1 --batch-size 64 --max-cells 512 --inr-dir /tmp/bc_inr_smoke --out-dir /tmp/bc_bcam_smoke
bash scripts/reproduce_bc_xenium_cluster.sh --max-cells 2000 --inr-dir /tmp/bc_inr_smoke --bcam-dir /tmp/bc_bcam_smoke --vis-dir /tmp/bc_cluster_smoke
```

## Verified in this workspace

Verified:

- Dry-run for the new UNI / BCAM / INR / cluster shell entrypoints
- Small breast cancer end-to-end smoke test
- Figure export using repository-local outputs

Not verified in this turn:

- Full-cell UNI extraction from scratch, because weights are not redistributed here
- Full 167,780-cell retraining for INR and BCAM in this turn
