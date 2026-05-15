from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch


def load_dlpfc_slice(h5ad_path: str, truth_path: str, use_all_entries: bool = True) -> dict:
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(h5ad_path)
    if not os.path.exists(truth_path):
        raise FileNotFoundError(truth_path)

    adata = sc.read_h5ad(h5ad_path)
    df_truth = pd.read_csv(truth_path, sep="\t", header=None, names=["barcode", "label"])
    truth_map = dict(zip(df_truth["barcode"], df_truth["label"]))

    obs_names = adata.obs_names.to_numpy()
    obs_trimmed = [b.split("-slice")[0] if "-slice" in b else b for b in obs_names]
    labels_all = np.array([truth_map.get(b, np.nan) for b in obs_trimmed])
    adata.obs["GroundTruth"] = labels_all

    x_coords = adata.obs["array_row"].to_numpy().astype(int)
    y_coords = adata.obs["array_col"].to_numpy().astype(int)
    x0 = x_coords - x_coords.min()
    y0 = y_coords - y_coords.min()
    n_x = int(x0.max()) + 1
    n_y = int(y0.max()) + 1

    raw = adata.X
    if sp.issparse(raw):
        raw = raw.tocoo()
        spot_idx, gene_idx, values = raw.row, raw.col, raw.data.astype(np.float32)
    else:
        spot_idx, gene_idx = np.nonzero(raw)
        values = raw[spot_idx, gene_idx].astype(np.float32)

    expr_3d = np.zeros((adata.n_vars, n_x, n_y), dtype=np.float32)
    expr_3d[gene_idx, x0[spot_idx], y0[spot_idx]] = values

    if use_all_entries:
        tissue_mask = np.zeros((n_x, n_y), dtype=bool)
        tissue_mask[x0, y0] = True
        sub = expr_3d[:, tissue_mask]
        sub[sub == 0] = -1.0
        expr_3d[:, tissue_mask] = sub

    expr_tensor = torch.from_numpy(expr_3d).to_sparse()
    N = n_x * n_y
    A_xy = (
        np.kron(np.eye(n_x, k=-1) + np.eye(n_x, k=1), np.eye(n_y))
        + np.kron(np.eye(n_x), np.eye(n_y, k=-1) + np.eye(n_y, k=1))
    )
    aligned = x0 * n_y + y0
    empty_idx = np.setdiff1d(np.arange(N), aligned)
    A_xy[empty_idx, :] = 0
    A_xy[:, empty_idx] = 0

    grid_labels = np.full(N, np.nan, dtype=object)
    grid_labels[aligned] = labels_all
    tissue_flat = np.sort(aligned)
    coords_mapping = np.column_stack([tissue_flat // n_y, tissue_flat % n_y]).astype(int)

    flat_to_orig = {}
    for xi, yi, xc, yc in zip(x0, y0, x_coords, y_coords):
        flat_to_orig[xi * n_y + yi] = (xc, yc)
    spatial_coords = np.array([flat_to_orig[f] for f in tissue_flat])
    labels_gt = grid_labels[tissue_flat]

    return {
        "expr_tensor": expr_tensor,
        "gene_names": adata.var_names.to_numpy(),
        "n_x": n_x,
        "n_y": n_y,
        "coords_mapping": coords_mapping,
        "spatial_coords": spatial_coords,
        "labels_gt": labels_gt,
        "A_xy": A_xy,
    }


def load_simulation_slice(h5ad_path: str, layer_name: str) -> dict:
    adata = sc.read_h5ad(h5ad_path)
    input_data = adata.layers[layer_name]
    original_data = adata.layers["original_norm"]

    if sp.issparse(input_data):
        input_data = input_data.toarray()
    if sp.issparse(original_data):
        original_data = original_data.toarray()

    array_row = adata.obs["array_row"].values.astype(int)
    array_col = adata.obs["array_col"].values.astype(int)
    n_x = int(array_row.max()) + 1
    n_y = int(array_col.max()) + 1

    n_g = adata.n_vars
    expr_3d = np.zeros((n_g, n_x, n_y), dtype=np.float32)
    for i in range(adata.n_obs):
        expr_3d[:, array_row[i], array_col[i]] = input_data[i]

    return {
        "expr_tensor": torch.from_numpy(expr_3d).to_sparse(),
        "gene_names": adata.var_names.to_numpy(),
        "n_x": n_x,
        "n_y": n_y,
        "coords_mapping": np.column_stack([array_row, array_col]).astype(int),
        "expr_original": original_data.astype(np.float32),
        "adata": adata,
    }

