from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from repro_st_aster.common import ensure_dir, save_json, seed_everything
from repro_st_aster.uni_bcam.bcam_core import BCAM


def main() -> int:
    parser = argparse.ArgumentParser(description="ASTER-SC step 2: BCAM on INR output and UNI features.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_input"))
    parser.add_argument("--inr-dir", type=Path, default=Path("preprocess_data/bc_xenium/inr_output"))
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_output"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    gene_expr_inr = np.load(args.inr_dir / "inr_reconstructed_expression.npy")
    gene_expr_raw = np.load(args.data_dir / "gene_expression_normalized.npy")
    uni2_feat = np.load(args.data_dir / "uni2_features_per_cell.npy")
    coords = np.load(args.data_dir / "cell_coords_standardized.npy")
    if args.max_cells is not None:
        keep = min(args.max_cells, gene_expr_raw.shape[0])
        gene_expr_inr = gene_expr_inr[:keep]
        gene_expr_raw = gene_expr_raw[:keep]
        uni2_feat = uni2_feat[:keep]
        coords = coords[:keep]

    if args.dry_run:
        print(gene_expr_inr.shape, gene_expr_raw.shape, uni2_feat.shape, coords.shape)
        return 0

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    ensure_dir(args.out_dir)
    n_cells, n_genes = gene_expr_raw.shape
    _, feat_dim = uni2_feat.shape
    nbrs = NearestNeighbors(n_neighbors=args.k_neighbors + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    knn_indices = indices[:, 1:]

    class SpatialDataset(Dataset):
        def __init__(self, gene_inr, gene_raw, hist, neighbors):
            self.gene_inr = torch.tensor(gene_inr, dtype=torch.float32)
            self.gene_raw = torch.tensor(gene_raw, dtype=torch.float32)
            self.hist = torch.tensor(hist, dtype=torch.float32)
            self.neighbors = torch.tensor(neighbors, dtype=torch.long)

        def __len__(self):
            return len(self.gene_raw)

        def __getitem__(self, idx):
            return {
                "gene_inr": self.gene_inr[idx],
                "gene_raw": self.gene_raw[idx],
                "hist": self.hist[idx],
                "neighbors": self.neighbors[idx],
            }

    idx = np.arange(n_cells)
    np.random.shuffle(idx)
    n_val = max(1, int(n_cells * 0.05))
    idx_val, idx_train = idx[:n_val], idx[n_val:]
    ds_train = SpatialDataset(gene_expr_inr[idx_train], gene_expr_raw[idx_train], uni2_feat[idx_train], knn_indices[idx_train])
    ds_val = SpatialDataset(gene_expr_inr[idx_val], gene_expr_raw[idx_val], uni2_feat[idx_val], knn_indices[idx_val])
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    dl_all = DataLoader(SpatialDataset(gene_expr_inr, gene_expr_raw, uni2_feat, knn_indices), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCAM(n_genes, feat_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    all_gene = torch.tensor(gene_expr_inr, dtype=torch.float32, device=device)
    all_hist = torch.tensor(uni2_feat, dtype=torch.float32, device=device)

    best_val = float("inf")
    for _epoch in range(args.epochs):
        model.train()
        for batch in dl_train:
            gene = batch["gene_inr"].to(device)
            target = batch["gene_raw"].to(device)
            hist = batch["hist"].to(device)
            neighbors = batch["neighbors"].to(device)
            _, recon = model(gene, hist, neighbors, all_gene, all_hist)
            loss = loss_fn(recon, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in dl_val:
                gene = batch["gene_inr"].to(device)
                target = batch["gene_raw"].to(device)
                hist = batch["hist"].to(device)
                neighbors = batch["neighbors"].to(device)
                _, recon = model(gene, hist, neighbors, all_gene, all_hist)
                val_losses.append(loss_fn(recon, target).item())
        val = float(np.mean(val_losses))
        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), args.out_dir / "bcam_best.pth")

    model.load_state_dict(torch.load(args.out_dir / "bcam_best.pth", map_location=device))
    torch.save(model.state_dict(), args.out_dir / "bcam_final.pth")

    latents, recons = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl_all, desc="extract"):
            gene = batch["gene_inr"].to(device)
            hist = batch["hist"].to(device)
            neighbors = batch["neighbors"].to(device)
            latent, recon = model(gene, hist, neighbors, all_gene, all_hist)
            latents.append(latent.cpu().numpy())
            recons.append(recon.cpu().numpy())
    np.save(args.out_dir / "fusion_latent_512.npy", np.vstack(latents))
    np.save(args.out_dir / "fusion_predicted_expression.npy", np.vstack(recons))
    save_json(args.out_dir / "training_info.json", {"n_cells": int(n_cells), "n_genes": int(n_genes), "best_val_loss": float(best_val)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
