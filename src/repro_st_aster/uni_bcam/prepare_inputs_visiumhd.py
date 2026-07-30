"""Build BCAM/FusionNet inputs from a raw Visium HD bundle (colorectal cancer, Fig. 4).

Differs from ``bcam_prepare_inputs.py`` (single-cell Xenium) in four ways:

1. expression comes from an MTX bundle plus ``tissue_positions.parquet`` rather than
   an h5 plus a coordinate CSV;
2. coordinates are mapped into standardized H&E space via the cropping metadata;
3. genes are filtered and then reduced to the top spatially variable genes by
   Moran's I, because the panel is whole-transcriptome;
4. spots are restricted to those falling inside the standardized H&E image, which is
   what gives a UNI-2 superpixel to every retained spot.

Memory note: the gene filter and normalization run while the matrix is still sparse,
Moran's I densifies only its subsample, and the full matrix is densified only after
slicing to the selected genes. Densifying before the gene slice would need ~39 GB.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

from repro_st_aster.common import ensure_dir, require_inputs, save_json
from repro_st_aster.uni_bcam.prepare_common import (
    inside_standardized_image,
    load_he_standardization,
    map_to_superpixel,
    select_svg_morans_i,
    standardize_coords,
)

# Colorectal Visium HD used a 14-px superpixel step, matching the true grid stride
# (224-px tiles -> 16x16 patches of 14 px). See the README "Deviations" section.
DEFAULT_SUPERPIXEL_STRIDE = 14

# Mitochondrial / ribosomal / housekeeping genes excluded before SVG selection.
HOUSEKEEPING_GENES = ["Actb", "Gapdh", "B2m", "Eef2", "Ppia"]


def filter_genes_basic(adata):
    mito = adata.var_names.str.upper().str.startswith("MT-")
    ribo = adata.var_names.str.match(r"^(Rps|Rpl|Mrp)")
    housekeeping = adata.var_names.isin(HOUSEKEEPING_GENES)
    return adata[:, ~(mito | ribo | housekeeping)].copy()


def prepare_inputs_hd(
    mtx_dir: Path,
    positions_parquet: Path,
    he_metadata: Path,
    uni_feature_path: Path,
    out_dir: Path,
    top_svg: int = 2500,
    svg_k: int = 15,
    svg_subsample: int = 12000,
    superpixel_stride: int = DEFAULT_SUPERPIXEL_STRIDE,
    max_cells: Optional[int] = None,
    seed: int = 42,
) -> dict:
    require_inputs(
        [
            mtx_dir / "barcodes.tsv.gz",
            mtx_dir / "features.tsv.gz",
            mtx_dir / "matrix.mtx.gz",
            positions_parquet,
            he_metadata,
            uni_feature_path,
        ],
        "crc_visiumhd",
    )
    ensure_dir(out_dir)

    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", make_unique=True)
    n_bins_raw = int(adata.n_obs)

    pos = pd.read_parquet(positions_parquet)
    pos = pos[pos["in_tissue"] == 1].copy()
    pos["barcode"] = pos["barcode"].astype(str)
    pos = pos.set_index("barcode")

    # Visium HD obs_names may or may not carry the -1 suffix; keep the better match.
    if adata.obs_names.isin(pos.index).mean() < adata.obs_names.str.replace("-1", "", regex=False).isin(pos.index).mean():
        adata.obs_names = adata.obs_names.str.replace("-1", "", regex=False)
    common = adata.obs_names.intersection(pos.index)
    adata = adata[common].copy()
    pos = pos.loc[adata.obs_names]

    he_meta = load_he_standardization(he_metadata)
    coords = standardize_coords(pos, he_meta)

    adata = filter_genes_basic(adata)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # SVG selection runs on the full in-tissue set, before the in-image filter
    # below, which is the order the published run used -- selecting after the
    # filter would draw a different Moran's I subsample and hence a different
    # gene set. Only the subsampled rows are densified (~0.8 GB).
    n_spots = int(adata.n_obs)
    if n_spots > svg_subsample:
        np.random.seed(seed)
        sub_idx = np.random.choice(n_spots, svg_subsample, replace=False)
    else:
        sub_idx = np.arange(n_spots)
    x_sub = adata.X[sub_idx]
    x_sub = x_sub.toarray().astype(np.float32) if issparse(x_sub) else np.asarray(x_sub, dtype=np.float32)
    # Moran's I depends only on the KNN structure, which a uniform rescale and an
    # axis swap leave unchanged, so standardized coordinates give the same graph as
    # the full-resolution pixel coordinates the original run used.
    # subsample=len(sub_idx) so select_svg_morans_i uses these rows as given.
    svg_idx, morans_i = select_svg_morans_i(
        x_sub,
        coords[sub_idx],
        k=svg_k,
        top_n=min(top_svg, adata.n_vars),
        subsample=len(sub_idx),
        seed=seed,
    )
    gene_names = adata.var_names.to_numpy()[svg_idx]

    # Keep only spots that land inside the standardized H&E, i.e. that have a
    # UNI-2 superpixel. This is the filter behind the published row count.
    inside = inside_standardized_image(coords, he_meta)
    n_dropped_out_of_image = int((~inside).sum())
    adata = adata[inside].copy()
    coords = coords[inside]

    # Densify only after slicing to the selected genes.
    x_svg = adata.X[:, svg_idx]
    gene_expr = x_svg.toarray().astype(np.float32) if issparse(x_svg) else np.asarray(x_svg, dtype=np.float32)
    cell_ids = adata.obs_names.to_numpy()

    if max_cells is not None:
        keep = min(max_cells, gene_expr.shape[0])
        gene_expr = gene_expr[:keep]
        coords = coords[:keep]
        cell_ids = cell_ids[:keep]

    superpixel_features = np.load(uni_feature_path, mmap_mode="r")
    grid_h, grid_w, feat_dim = superpixel_features.shape
    sp_i, sp_j, valid = map_to_superpixel(coords, (grid_h, grid_w), superpixel_stride)
    n_dropped_out_of_grid = int((~valid).sum())
    if n_dropped_out_of_grid:
        gene_expr = gene_expr[valid]
        coords = coords[valid]
        cell_ids = cell_ids[valid]
        sp_i, sp_j = sp_i[valid], sp_j[valid]
    uni2_feat = np.asarray(superpixel_features[sp_i, sp_j, :], dtype=np.float32)

    np.save(out_dir / "gene_expression_normalized.npy", gene_expr)
    np.save(out_dir / "uni2_features_per_cell.npy", uni2_feat)
    np.save(out_dir / "cell_coords_standardized.npy", coords.astype(np.float32))
    np.save(out_dir / "gene_names.npy", gene_names)
    np.save(out_dir / "cell_ids.npy", cell_ids)
    np.save(out_dir / "superpixel_mapping.npy", np.column_stack([sp_i, sp_j]))
    pd.DataFrame(
        {"svg_rank": np.arange(len(svg_idx)), "gene_name": gene_names, "morans_i": morans_i[svg_idx]}
    ).to_csv(out_dir / "svg_genes_info.csv", index=False)

    meta = {
        "n_cells": int(gene_expr.shape[0]),
        "n_genes": int(gene_expr.shape[1]),
        "n_bins_raw": n_bins_raw,
        "n_dropped_out_of_image": n_dropped_out_of_image,
        "n_dropped_out_of_grid": n_dropped_out_of_grid,
        "uni2_feature_dim": int(feat_dim),
        "uni2_grid_shape": [int(grid_h), int(grid_w)],
        "superpixel_size_pixels": int(superpixel_stride),
        "svg_top_n": int(top_svg),
        "svg_moran_k": int(svg_k),
        "svg_subsample": int(svg_subsample),
        "normalization": "scanpy.normalize_total(target_sum=1e4) + log1p",
    }
    save_json(out_dir / "preprocessing_metadata.json", meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    """Entry point; ``argv`` lets a notebook call this like a function."""
    parser = argparse.ArgumentParser(description="Prepare fusion inputs from a raw Visium HD bundle.")
    parser.add_argument(
        "--mtx-dir",
        type=Path,
        default=Path("raw_data/crc_visiumhd/binned_outputs/square_008um/filtered_feature_bc_matrix"),
    )
    parser.add_argument(
        "--positions-parquet",
        type=Path,
        default=Path("raw_data/crc_visiumhd/binned_outputs/square_008um/spatial/tissue_positions.parquet"),
    )
    parser.add_argument("--he-metadata", type=Path, default=Path("raw_data/crc_visiumhd/metadata.json"))
    parser.add_argument(
        "--uni-feature-path", type=Path, default=Path("preprocess_data/crc_visiumhd/uni/superpixel_features.npy")
    )
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/crc_visiumhd/bcam_input"))
    parser.add_argument("--top-svg", type=int, default=2500)
    parser.add_argument("--svg-k", type=int, default=15)
    parser.add_argument("--svg-subsample", type=int, default=12000)
    parser.add_argument("--superpixel-stride", type=int, default=DEFAULT_SUPERPIXEL_STRIDE)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(args.mtx_dir)
        print(args.positions_parquet)
        print(args.he_metadata)
        print(args.uni_feature_path)
        print(f"output -> {args.out_dir}")
        return 0

    meta = prepare_inputs_hd(
        args.mtx_dir,
        args.positions_parquet,
        args.he_metadata,
        args.uni_feature_path,
        args.out_dir,
        top_svg=args.top_svg,
        svg_k=args.svg_k,
        svg_subsample=args.svg_subsample,
        superpixel_stride=args.superpixel_stride,
        max_cells=args.max_cells,
        seed=args.seed,
    )
    print(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
