from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from repro_st_aster.common import ensure_dir, save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="ASTER-SC step 3: cluster and visualize.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_input"))
    parser.add_argument("--bcam-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_output"))
    parser.add_argument("--inr-dir", type=Path, default=Path("preprocess_data/bc_xenium/inr_output"))
    parser.add_argument("--vis-dir", type=Path, default=Path("preprocess_data/bc_xenium/viz"))
    parser.add_argument("--k", type=int, default=17)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    latent_path = args.bcam_dir / "fusion_latent_512.npy"
    recon_path = args.bcam_dir / "fusion_predicted_expression.npy"
    inr_path = args.inr_dir / "inr_reconstructed_expression.npy"
    coords_path = args.data_dir / "cell_coords_standardized.npy"
    gene_path = args.data_dir / "gene_names.npy"

    if args.dry_run:
        for path in [latent_path, recon_path, inr_path, coords_path, gene_path]:
            print(path)
        return 0

    ensure_dir(args.vis_dir)
    latent = np.load(latent_path)
    recon = np.load(recon_path)
    inr_expr = np.load(inr_path)
    coords = np.load(coords_path)
    gene_names = np.load(gene_path, allow_pickle=True)
    if args.max_cells is not None:
        keep = min(args.max_cells, latent.shape[0])
        latent = latent[:keep]
        recon = recon[:keep]
        inr_expr = inr_expr[:keep]
        coords = coords[:keep]
    labels = KMeans(n_clusters=args.k, random_state=42, n_init=20).fit_predict(latent)
    np.save(args.vis_dir / f"labels_fusion_K{args.k}.npy", labels)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=labels, s=1, cmap="tab20", alpha=0.8)
    axes[0, 0].set_title(f"ASTER-SC Fusion Clustering (K={args.k})")
    axes[0, 0].axis("off")
    pca = PCA(n_components=2).fit_transform(latent)
    axes[0, 1].scatter(pca[:, 0], pca[:, 1], c=labels, s=1, cmap="tab20", alpha=0.5)
    axes[0, 1].set_title("Latent PCA")
    example_idx = 0
    axes[1, 0].scatter(coords[:, 0], coords[:, 1], c=inr_expr[:, example_idx], s=1, cmap="turbo", alpha=0.8)
    axes[1, 0].set_title(f"{gene_names[example_idx]} INR")
    axes[1, 1].scatter(coords[:, 0], coords[:, 1], c=recon[:, example_idx], s=1, cmap="turbo", alpha=0.8)
    axes[1, 1].set_title(f"{gene_names[example_idx]} BCAM")
    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.vis_dir / f"clustering_K{args.k}_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    save_json(args.vis_dir / "cluster_summary.json", {"k": args.k, "n_cells": int(latent.shape[0])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
