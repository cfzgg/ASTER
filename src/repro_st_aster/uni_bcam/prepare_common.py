"""Preprocessing helpers shared by the BCAM input builders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def select_svg_morans_i(
    X: np.ndarray,
    coords: np.ndarray,
    k: int = 15,
    top_n: int = 2500,
    subsample: int = 12000,
    seed: int = 42,
):
    """Select the top ``top_n`` spatially variable genes by Moran's I.

    Moran's I is computed on a random subsample of spots (``subsample``) using a
    ball-tree KNN graph with self excluded.

    .. note::
       The subsample uses the legacy global RNG (``np.random.seed`` +
       ``np.random.choice``) on purpose: this reproduces the gene set of the
       published colorectal run. ``np.random.default_rng`` draws a *different*
       subsample and therefore selects a different gene set.

    Returns ``(selected_indices, morans_i)``.
    """
    n = X.shape[0]
    if n > subsample:
        np.random.seed(seed)
        idx_sp = np.random.choice(n, subsample, replace=False)
    else:
        idx_sp = np.arange(n)

    xs_sub = X[idx_sp]
    coords_sub = coords[idx_sp]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(idx_sp)), algorithm="ball_tree").fit(coords_sub)
    ind = nbrs.kneighbors(return_distance=False)[:, 1:]

    xs = xs_sub - xs_sub.mean(0, keepdims=True)
    denom = (xs ** 2).sum(0) + 1e-12
    num = np.zeros(xs.shape[1], dtype=np.float64)
    for i in range(len(idx_sp)):
        num += xs[i] * xs[ind[i]].sum(0)
    morans_i = (len(idx_sp) / (k * len(idx_sp))) * (num / denom)
    return np.argsort(morans_i)[::-1][:top_n], morans_i


def load_he_standardization(metadata_path: Path) -> dict:
    """Read the H&E standardization metadata written by the cropping workflow."""
    with open(metadata_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    height, width = meta["standardization"]["shape_after_padding"][:2]
    return {
        "x_min": float(meta["tissue_bbox"]["x_min"]),
        "y_min": float(meta["tissue_bbox"]["y_min"]),
        "scale_factor": float(meta["standardization"]["scale_factor"]),
        "image_height": int(height),
        "image_width": int(width),
    }


def standardize_coords(pos: pd.DataFrame, he_meta: dict) -> np.ndarray:
    """Map Visium HD full-resolution pixel coordinates into standardized H&E space.

    ``x_std = (pxl_col_in_fullres - x_min) * scale_factor`` and likewise for y with
    ``pxl_row_in_fullres``. No flips or transposes.
    """
    x_std = (pos["pxl_col_in_fullres"].to_numpy(dtype=np.float64) - he_meta["x_min"]) * he_meta["scale_factor"]
    y_std = (pos["pxl_row_in_fullres"].to_numpy(dtype=np.float64) - he_meta["y_min"]) * he_meta["scale_factor"]
    return np.column_stack([x_std, y_std])


def inside_standardized_image(coords: np.ndarray, he_meta: dict) -> np.ndarray:
    """Boolean mask of spots whose standardized coordinates fall inside the H&E image.

    This is the filter that reduces the colorectal in-tissue set to the spots that
    have a UNI-2 superpixel, i.e. the row set the published run works on.
    """
    return (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < he_meta["image_width"])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < he_meta["image_height"])
    )


def map_to_superpixel(coords: np.ndarray, grid_shape, stride: int):
    """Map standardized coordinates to UNI-2 superpixel (row, col) indices.

    Returns ``(sp_i, sp_j, valid_mask)`` where ``valid_mask`` marks indices inside
    the feature grid. ``stride`` is the pixel step per superpixel; see the
    per-dataset note in the README (colorectal uses 14, the single-cell workflows
    use 16).
    """
    grid_h, grid_w = grid_shape[0], grid_shape[1]
    sp_i = (coords[:, 1] // stride).astype(int)
    sp_j = (coords[:, 0] // stride).astype(int)
    valid = (sp_i >= 0) & (sp_i < grid_h) & (sp_j >= 0) & (sp_j < grid_w)
    return sp_i, sp_j, valid
