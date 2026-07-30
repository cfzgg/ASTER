"""ASTER-SC fusion stage for Visium HD (colorectal cancer, Fig. 4).

Trains :class:`repro_st_aster.aster_sc.fusion_core.FusionNet` to reconstruct the raw
lognorm expression of each bin from the INR reconstruction plus UNI-2 morphology over
its K-nearest neighbourhood, then writes the 512-d clustering latent.

Outputs use the same filenames as ``bcam.py`` so that ``cluster_visualize.py`` can
consume either workflow without changes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from repro_st_aster.common import build_knn, ensure_dir, require_inputs, save_json, seed_everything

LATENT_DIM = 512


def main(argv: list[str] | None = None) -> int:
    """Entry point; ``argv`` lets a notebook call this like a function."""
    parser = argparse.ArgumentParser(description="Train FusionNet on INR output and UNI-2 features (Visium HD).")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/crc_visiumhd/bcam_input"))
    parser.add_argument("--inr-dir", type=Path, default=Path("preprocess_data/crc_visiumhd/inr_output"))
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/crc_visiumhd/bcam_output"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k-neighbors", type=int, default=8, help="Neighbourhood size including the centre bin.")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
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

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    from repro_st_aster.aster_sc.fusion_core import FusionNet

    require_inputs([inr_path, raw_path, uni_path, coords_path], "crc_visiumhd")

    seed_everything(args.seed)
    ensure_dir(args.out_dir)
    ckpt_dir = ensure_dir(args.out_dir / "checkpoints")

    inr_expr = np.load(inr_path)
    raw_expr = np.load(raw_path)
    uni_feat = np.load(uni_path)
    coords = np.load(coords_path)
    if args.max_cells is not None:
        keep = min(args.max_cells, raw_expr.shape[0])
        inr_expr = inr_expr[:keep]
        raw_expr = raw_expr[:keep]
        uni_feat = uni_feat[:keep]
        coords = coords[:keep]
    n_spot, n_gene = raw_expr.shape
    uni_dim = uni_feat.shape[1]

    # FusionNet reads neighbour 0 as the centre bin, so self must be in the graph.
    knn = build_knn(coords, args.k_neighbors, include_self=True)
    probe = np.linspace(0, n_spot - 1, num=min(64, n_spot)).astype(int)
    assert (knn[probe, 0] == probe).all(), "KNN graph must include self as neighbour 0"

    class FusionDataset(Dataset):
        def __len__(self):
            return n_spot

        def __getitem__(self, idx):
            neigh = knn[idx]
            return {
                "uni2": torch.tensor(uni_feat[neigh], dtype=torch.float32),
                "inr": torch.tensor(inr_expr[neigh], dtype=torch.float32),
                "raw": torch.tensor(raw_expr[idx], dtype=torch.float32),
                "center_idx": idx,
            }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNet(uni_dim, n_gene, n_gene, embed_dim=args.embed_dim, num_heads=args.num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset = FusionDataset()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loader:
            # NOTE: FusionNet returns (pred, latent) -- the reverse of BCAM.
            pred, _ = model(batch["uni2"].to(device), batch["inr"].to(device))
            loss = F.smooth_l1_loss(pred, batch["raw"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"  Fusion ep {epoch:03d} | loss={running / max(len(loader), 1):.4f}")
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
            torch.save(model.state_dict(), ckpt_dir / f"fusion_epoch_{epoch:03d}.pt")

    # Latent extraction must not shuffle: rows are scattered back by center_idx.
    model.eval()
    latent_all = np.zeros((n_spot, LATENT_DIM), dtype=np.float32)
    pred_all = np.zeros((n_spot, n_gene), dtype=np.float32)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    with torch.no_grad():
        for batch in eval_loader:
            pred, latent = model(batch["uni2"].to(device), batch["inr"].to(device))
            assert latent.shape[1] == LATENT_DIM, f"expected a {LATENT_DIM}-d latent, got {latent.shape[1]}"
            idx = batch["center_idx"].numpy()
            latent_all[idx] = latent.cpu().numpy()
            pred_all[idx] = pred.cpu().numpy()

    np.save(args.out_dir / "fusion_latent_512.npy", latent_all)
    np.save(args.out_dir / "fusion_predicted_expression.npy", pred_all)
    save_json(
        args.out_dir / "training_info.json",
        {
            "n_cells": int(n_spot),
            "n_genes": int(n_gene),
            "uni_dim": int(uni_dim),
            "latent_dim": LATENT_DIM,
            "epochs": int(args.epochs),
            "k_neighbors": int(args.k_neighbors),
            "embed_dim": int(args.embed_dim),
            "num_heads": int(args.num_heads),
            "loss": "smooth_l1 (pred vs raw lognorm)",
        },
    )
    print(f"fusion latent: {latent_all.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
