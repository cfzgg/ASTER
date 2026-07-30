from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from repro_st_aster.common import ensure_dir, save_json, seed_everything
from repro_st_aster.uni_bcam.bcam_core import BCAM


def build_dataset(data_dir: Path, k_neighbors: int):
    gene_expr = np.load(data_dir / "gene_expression_normalized.npy")
    uni2_feat = np.load(data_dir / "uni2_features_per_cell.npy")
    coords = np.load(data_dir / "cell_coords_standardized.npy")
    gene_names = np.load(data_dir / "gene_names.npy", allow_pickle=True)

    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    return gene_expr, uni2_feat, coords, gene_names, indices[:, 1:]


def main(argv: list[str] | None = None) -> int:
    """Entry point; ``argv`` lets a notebook call this like a function."""
    parser = argparse.ArgumentParser(description="Train BCAM on raw gene expression and UNI features.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_input"))
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_output"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        for name in [
            "gene_expression_normalized.npy",
            "uni2_features_per_cell.npy",
            "cell_coords_standardized.npy",
            "gene_names.npy",
        ]:
            print(args.data_dir / name)
        print(f"output -> {args.out_dir}")
        return 0

    seed_everything(args.seed)
    gene_expr, uni2_feat, _coords, gene_names, knn = build_dataset(args.data_dir, args.k_neighbors)
    if args.max_cells is not None:
        keep = min(args.max_cells, gene_expr.shape[0])
        gene_expr = gene_expr[:keep]
        uni2_feat = uni2_feat[:keep]
        gene_names = gene_names
        knn = np.clip(knn[:keep], 0, keep - 1)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    n_cells, n_genes = gene_expr.shape
    _, feat_dim = uni2_feat.shape
    ensure_dir(args.out_dir)

    class SpatialDataset(Dataset):
        def __init__(self, gene, hist, neighbors):
            self.gene = torch.tensor(gene, dtype=torch.float32)
            self.hist = torch.tensor(hist, dtype=torch.float32)
            self.neighbors = torch.tensor(neighbors, dtype=torch.long)

        def __len__(self):
            return len(self.gene)

        def __getitem__(self, idx):
            return {"gene": self.gene[idx], "hist": self.hist[idx], "neighbors": self.neighbors[idx]}

    idx = np.arange(n_cells)
    np.random.shuffle(idx)
    n_val = max(1, int(n_cells * 0.05))
    idx_val, idx_train = idx[:n_val], idx[n_val:]

    ds_train = SpatialDataset(gene_expr[idx_train], uni2_feat[idx_train], knn[idx_train])
    ds_val = SpatialDataset(gene_expr[idx_val], uni2_feat[idx_val], knn[idx_val])
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    dl_all = DataLoader(SpatialDataset(gene_expr, uni2_feat, knn), batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BCAM(n_genes, feat_dim, args.hidden_dim, args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    all_gene = torch.tensor(gene_expr, dtype=torch.float32, device=device)
    all_hist = torch.tensor(uni2_feat, dtype=torch.float32, device=device)

    best_val = float("inf")
    for _epoch in range(args.epochs):
        model.train()
        for batch in dl_train:
            gene = batch["gene"].to(device)
            hist = batch["hist"].to(device)
            neighbors = batch["neighbors"].to(device)
            _, recon = model(gene, hist, neighbors, all_gene, all_hist)
            loss = loss_fn(recon, gene)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in dl_val:
                gene = batch["gene"].to(device)
                hist = batch["hist"].to(device)
                neighbors = batch["neighbors"].to(device)
                _, recon = model(gene, hist, neighbors, all_gene, all_hist)
                val_losses.append(loss_fn(recon, gene).item())
        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out_dir / "bcam_best.pth")

    model.load_state_dict(torch.load(args.out_dir / "bcam_best.pth", map_location=device))
    torch.save(model.state_dict(), args.out_dir / "bcam_final.pth")

    latents, recons = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl_all, desc="extract"):
            gene = batch["gene"].to(device)
            hist = batch["hist"].to(device)
            neighbors = batch["neighbors"].to(device)
            latent, recon = model(gene, hist, neighbors, all_gene, all_hist)
            latents.append(latent.cpu().numpy())
            recons.append(recon.cpu().numpy())
    np.save(args.out_dir / "fusion_latent_512.npy", np.vstack(latents))
    np.save(args.out_dir / "fusion_predicted_expression.npy", np.vstack(recons))
    save_json(
        args.out_dir / "training_info.json",
        {
            "n_cells": int(n_cells),
            "n_genes": int(n_genes),
            "feat_dim": int(feat_dim),
            "latent_dim": int(args.latent_dim),
            "best_val_loss": float(best_val),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
