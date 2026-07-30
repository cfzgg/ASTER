from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from repro_st_aster.common import ensure_dir, require_inputs, save_json

# Pixel step per superpixel used to map a cell onto the UNI-2 feature grid.
#
# The published single-cell runs (breast cancer Fig. 3, gastric cancer Fig. 5) used
# 16, which is the value kept here so those figures reproduce. Note that the feature
# grid itself has a true stride of 14 px (224-px tiles -> 16x16 patches of 14 px), so
# 16 addresses only the leading part of the grid. The Visium HD colorectal workflow
# used 14; see `prepare_inputs_visiumhd.py` and the README "Deviations" section.
DEFAULT_SUPERPIXEL_STRIDE = 16


def prepare_inputs(
    matrix_h5: Path,
    coord_csv: Path,
    uni_feature_path: Path,
    out_dir: Path,
    max_cells: Optional[int] = None,
    superpixel_stride: int = DEFAULT_SUPERPIXEL_STRIDE,
) -> dict:
    require_inputs([matrix_h5, coord_csv, uni_feature_path])
    ensure_dir(out_dir)
    adata = sc.read_10x_h5(matrix_h5)
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cell_coords = pd.read_csv(coord_csv)
    # mmap: the gastric feature grid is ~20 GB on disk and must not be read whole.
    superpixel_features = np.load(uni_feature_path, mmap_mode="r")
    grid_h, grid_w, feat_dim = superpixel_features.shape

    x_std = cell_coords["x_standardized"].to_numpy()
    y_std = cell_coords["y_standardized"].to_numpy()
    sp_i = (y_std // superpixel_stride).astype(int)
    sp_j = (x_std // superpixel_stride).astype(int)

    valid = (sp_i >= 0) & (sp_i < grid_h) & (sp_j >= 0) & (sp_j < grid_w)
    n_dropped = int((~valid).sum())
    if not np.all(valid):
        adata = adata[valid].copy()
        cell_coords = cell_coords[valid].reset_index(drop=True)
        sp_i = sp_i[valid]
        sp_j = sp_j[valid]
        x_std = x_std[valid]
        y_std = y_std[valid]

    if max_cells is not None:
        keep = min(max_cells, adata.n_obs)
        adata = adata[:keep].copy()
        cell_coords = cell_coords.iloc[:keep].reset_index(drop=True)
        sp_i = sp_i[:keep]
        sp_j = sp_j[:keep]
        x_std = x_std[:keep]
        y_std = y_std[:keep]

    gene_expr = adata.X.toarray() if issparse(adata.X) else np.asarray(adata.X)
    uni2_feat = np.asarray(superpixel_features[sp_i, sp_j, :], dtype=np.float32)

    np.save(out_dir / "gene_expression_normalized.npy", gene_expr)
    np.save(out_dir / "uni2_features_per_cell.npy", uni2_feat)
    np.save(out_dir / "cell_coords_standardized.npy", np.column_stack([x_std, y_std]))
    np.save(out_dir / "gene_names.npy", adata.var_names.to_numpy())
    np.save(out_dir / "cell_ids.npy", adata.obs_names.to_numpy())
    np.save(out_dir / "superpixel_mapping.npy", np.column_stack([sp_i, sp_j]))

    meta = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "uni2_feature_dim": int(feat_dim),
        "uni2_grid_shape": [int(grid_h), int(grid_w)],
        "superpixel_size_pixels": int(superpixel_stride),
        "n_dropped_out_of_grid": n_dropped,
        "normalization": "scanpy.normalize_total(target_sum=1e4) + log1p",
    }
    save_json(out_dir / "preprocessing_metadata.json", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    """Entry point; ``argv`` lets a notebook call this like a function."""
    parser = argparse.ArgumentParser(description="Prepare BCAM inputs from raw breast cancer Xenium data and UNI features.")
    parser.add_argument("--matrix-h5", type=Path, default=Path("raw_data/bc_xenium/cell_feature_matrix.h5"))
    parser.add_argument("--coord-csv", type=Path, default=Path("raw_data/bc_xenium/cell_coordinates.csv"))
    parser.add_argument("--uni-feature-path", type=Path, default=Path("preprocess_data/bc_xenium/uni/superpixel_features.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_input"))
    parser.add_argument(
        "--superpixel-stride",
        type=int,
        default=DEFAULT_SUPERPIXEL_STRIDE,
        help="Pixel step per UNI-2 superpixel when mapping cells (published single-cell runs used 16).",
    )
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(args.matrix_h5)
        print(args.coord_csv)
        print(args.uni_feature_path)
        print(f"output -> {args.out_dir}")
        return 0

    meta = prepare_inputs(
        args.matrix_h5,
        args.coord_csv,
        args.uni_feature_path,
        args.out_dir,
        args.max_cells,
        superpixel_stride=args.superpixel_stride,
    )
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
