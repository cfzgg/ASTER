from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from repro_st_aster.common import build_knn, ensure_dir, require_inputs, save_json, seed_everything
from repro_st_aster.uni_bcam.bcam_core import BCAM


def main(argv: list[str] | None = None) -> int:
    """Entry point; ``argv`` lets a notebook call this like a function."""
    parser = argparse.ArgumentParser(description="ASTER-SC step 2: BCAM on INR output and UNI features.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_input"))
    parser.add_argument("--inr-dir", type=Path, default=Path("preprocess_data/bc_xenium/inr_output"))
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_output"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k-neighbors", type=int, default=8)
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine_warm_restarts"],
        default="none",
        help="LR schedule; the gastric run used cosine_warm_restarts (T_0=10, T_mult=2, eta_min=1e-6).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    inr_path = args.inr_dir / "inr_reconstructed_expression.npy"
    raw_path = args.data_dir / "gene_expression_normalized.npy"
    uni_path = args.data_dir / "uni2_features_per_cell.npy"
    coords_path = args.data_dir / "cell_coords_standardized.npy"

    if args.dry_run:
        for path in [inr_path, raw_path, uni_path, coords_path]:
            print(path)
        print(f"output -> {args.out_dir}")
        return 0

    require_inputs([inr_path, raw_path, uni_path, coords_path])

    seed_everything(args.seed)
    gene_expr_inr = np.load(inr_path)
    gene_expr_raw = np.load(raw_path)
    uni2_feat = np.load(uni_path)
    coords = np.load(coords_path)
    if args.max_cells is not None:
        keep = min(args.max_cells, gene_expr_raw.shape[0])
        gene_expr_inr = gene_expr_inr[:keep]
        gene_expr_raw = gene_expr_raw[:keep]
        uni2_feat = uni2_feat[:keep]
        coords = coords[:keep]

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    ensure_dir(args.out_dir)
    n_cells, n_genes = gene_expr_raw.shape
    _, feat_dim = uni2_feat.shape
    # BCAM neighbourhoods exclude self; LocalKNNAttention re-adds the query token.
    knn_indices = build_knn(coords, args.k_neighbors, include_self=False)

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
    sched = None
    if args.scheduler == "cosine_warm_restarts":
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
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
        if sched is not None:
            sched.step()
        print(f"epoch {_epoch + 1:03d} val={val:.4f} lr={opt.param_groups[0]['lr']:.3e}")
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
    save_json(
        args.out_dir / "training_info.json",
        {
            "n_cells": int(n_cells),
            "n_genes": int(n_genes),
            "feat_dim": int(feat_dim),
            "epochs": int(args.epochs),
            "k_neighbors": int(args.k_neighbors),
            "scheduler": args.scheduler,
            "best_val_loss": float(best_val),
            "using_inr": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
