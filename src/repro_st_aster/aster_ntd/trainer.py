from __future__ import annotations

import gc
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

from .model import ASTER_NTD
from .utils import MSE


class ASTER:
    def __init__(self, expr_tensor: torch.Tensor, gene_names, n_x: int, n_y: int, coords_mapping: np.ndarray):
        self.expr_tensor = expr_tensor
        self.gene_names = np.asarray(gene_names)
        self.n_g = len(self.gene_names)
        self.n_x = n_x
        self.n_y = n_y
        self.coords_mapping = np.asarray(coords_mapping, dtype=int)

    def _split_data(self, ratio: float = 0.1, random_state: int = 12346):
        index = self.expr_tensor.indices().numpy().T
        ss = ShuffleSplit(n_splits=1, test_size=ratio, random_state=random_state)
        train_idx, val_idx = next(ss.split(index))

        def _pack(idx):
            expr = self.expr_tensor.values()[idx].clone()
            expr[expr < 0] = 0.0
            coords = self.expr_tensor.indices()[:, idx]
            flat = coords[0] * self.n_x * self.n_y + coords[1] * self.n_y + coords[2]
            return expr, flat

        return (*_pack(train_idx), *_pack(val_idx))

    def _train_step(self, expr, flat_idx, g_idx, X_in, Y_in) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        expr = expr.to(self.device, dtype=torch.float32)
        expr_hat = torch.flatten(self.model(self.core, g_idx, X_in, Y_in))[flat_idx]
        loss = F.mse_loss(expr, expr_hat, reduction="sum")
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @torch.no_grad()
    def _validate(self, expr, flat_idx, g_idx, X_in, Y_in) -> float:
        self.model.eval()
        expr_hat = torch.flatten(self.model(self.core, g_idx, X_in, Y_in))[flat_idx].cpu().numpy()
        return MSE(expr.numpy(), expr_hat)

    @torch.no_grad()
    def _reconstruct(self, g_idx, X_in, Y_in) -> torch.Tensor:
        self.model.eval()
        return self.model(self.core, g_idx, X_in, Y_in).cpu()

    def fit(
        self,
        rank_x: int = 128,
        rank_y: int = 128,
        rank_g: int = 128,
        mid_feature: int = 200,
        lr: float = 0.001,
        max_epoch: int = 1200,
        save_dir: str = ".",
        verbose: bool = False,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_in = torch.arange(1, self.n_x + 1, dtype=torch.float32).reshape(-1, 1).to(self.device)
        Y_in = torch.arange(1, self.n_y + 1, dtype=torch.float32).reshape(-1, 1).to(self.device)
        g_idx = torch.arange(self.n_g, dtype=torch.long).to(self.device)

        self.core = nn.Parameter(torch.empty(rank_x, rank_y, rank_g, device=self.device, dtype=torch.float32))
        nn.init.xavier_uniform_(self.core)

        train_expr, train_idx, val_expr, val_idx = self._split_data()
        self.model = ASTER_NTD(self.n_g, rank_x, rank_y, rank_g, mid_feature).to(self.device)
        self.optimizer = torch.optim.Adam(list(self.model.parameters()) + [self.core], lr=lr)

        ckpt_model = os.path.join(save_dir, "aster_best_model.pt")
        ckpt_core = os.path.join(save_dir, "aster_best_core.pt")

        best_mse = np.inf
        pbar = tqdm(range(max_epoch), desc="ASTER", disable=not verbose)
        for _ in pbar:
            loss = self._train_step(train_expr, train_idx, g_idx, X_in, Y_in)
            mse = self._validate(val_expr, val_idx, g_idx, X_in, Y_in)
            if verbose:
                pbar.set_postfix({"loss": f"{loss:.4f}", "val_mse": f"{mse:.6f}"})
            if mse < best_mse:
                best_mse = mse
                torch.save(self.model.state_dict(), ckpt_model)
                torch.save(self.core.data, ckpt_core)
        pbar.close()

        self.model.load_state_dict(torch.load(ckpt_model, map_location=self.device))
        self.core = torch.load(ckpt_core, map_location=self.device)
        self.expr_tensor_hat = self._reconstruct(g_idx, X_in, Y_in)

    def get_imputed_matrix(self, gene_names=None):
        expr_3d = self.expr_tensor_hat.numpy()
        rows = self.coords_mapping[:, 0]
        cols = self.coords_mapping[:, 1]
        expr_mat = expr_3d[:, rows, cols].T
        if gene_names is None:
            return expr_mat, self.gene_names
        sel = np.array([g for g in gene_names if g in self.gene_names])
        if sel.size == 0:
            warnings.warn("No requested genes found; returning all genes.")
            return expr_mat, self.gene_names
        idx = np.array([np.where(self.gene_names == g)[0][0] for g in sel])
        return expr_mat[:, idx], sel

    def get_raw_matrix(self):
        expr_3d = self.expr_tensor.to_dense().numpy()
        expr_3d = np.clip(expr_3d, 0.0, None)
        rows = self.coords_mapping[:, 0]
        cols = self.coords_mapping[:, 1]
        return expr_3d[:, rows, cols].T, self.gene_names

    def clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

