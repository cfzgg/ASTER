from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from repro_st_aster.common import ensure_dir, get_palette, knn_gaussian_smooth, require_inputs, save_json


def main(argv: list[str] | None = None) -> int:
    """Entry point; ``argv`` lets a notebook call this like a function."""
    parser = argparse.ArgumentParser(description="ASTER-SC step 3: cluster and visualize.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_input"))
    parser.add_argument("--bcam-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_output"))
    parser.add_argument("--inr-dir", type=Path, default=Path("preprocess_data/bc_xenium/inr_output"))
    parser.add_argument("--vis-dir", type=Path, default=Path("preprocess_data/bc_xenium/viz"))
    parser.add_argument("--k", type=int, default=17)
    parser.add_argument(
        "--smooth-k",
        type=int,
        default=0,
        help="Gaussian KNN smoothing of the latent before KMeans; 0 disables it (colorectal run used 30).",
    )
    parser.add_argument("--kmeans-random-state", type=int, default=42)
    parser.add_argument(
        "--kmeans-n-init",
        default="20",
        help="KMeans n_init; an integer or 'auto' (colorectal run used auto, single-cell runs used 20).",
    )
    parser.add_argument(
        "--palette",
        default="none",
        help="Published palette name (e.g. fig5_gastric); 'none' uses the tab20 colormap, as Fig. 4a does.",
    )
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument("--point-alpha", type=float, default=0.8)
    parser.add_argument("--marker", default="o", help="Scatter marker; the Visium HD bin map uses 's'.")
    parser.add_argument("--invert-yaxis", action="store_true", help="Flip y to match the published Fig. 4a / 5a orientation.")
    parser.add_argument(
        "--extra-labels-name",
        default=None,
        help="Optional second filename for the labels, e.g. kmeans_labels_K15.npy to match the published artifact name.",
    )
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    latent_path = args.bcam_dir / "fusion_latent_512.npy"
    recon_path = args.bcam_dir / "fusion_predicted_expression.npy"
    inr_path = args.inr_dir / "inr_reconstructed_expression.npy"
    coords_path = args.data_dir / "cell_coords_standardized.npy"
    gene_path = args.data_dir / "gene_names.npy"

    if args.dry_run:
        for path in [latent_path, recon_path, inr_path, coords_path, gene_path]:
            print(path)
        print(f"output -> {args.vis_dir}")
        return 0

    n_init = args.kmeans_n_init if args.kmeans_n_init == "auto" else int(args.kmeans_n_init)

    require_inputs([latent_path, recon_path, inr_path, coords_path, gene_path])

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

    if args.smooth_k > 0:
        latent = knn_gaussian_smooth(latent, coords, k=args.smooth_k)
        np.save(args.bcam_dir / "fusion_latent_512_smooth.npy", latent)

    labels = KMeans(n_clusters=args.k, random_state=args.kmeans_random_state, n_init=n_init).fit_predict(latent)
    np.save(args.vis_dir / f"labels_fusion_K{args.k}.npy", labels)
    if args.extra_labels_name:
        np.save(args.vis_dir / args.extra_labels_name, labels)

    cmap = ListedColormap(get_palette(args.palette, args.k)) if args.palette != "none" else plt.get_cmap("tab20")

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes[0, 0].scatter(coords[:, 0], coords[:, 1], c=labels, s=1, cmap=cmap, alpha=0.8)
    axes[0, 0].set_title(f"ASTER-SC Fusion Clustering (K={args.k})")
    axes[0, 0].axis("off")
    pca = PCA(n_components=2).fit_transform(latent)
    axes[0, 1].scatter(pca[:, 0], pca[:, 1], c=labels, s=1, cmap=cmap, alpha=0.5)
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

    # Standalone domain map at publication geometry: figure aspect follows the
    # tissue aspect, and the point cloud is rasterized so the PDF stays editable.
    span = np.ptp(coords, axis=0)
    aspect = float(span[0] / max(span[1], 1e-8))
    panel_h = 8.0
    fig, ax = plt.subplots(figsize=(panel_h * aspect, panel_h))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        s=args.point_size,
        cmap=cmap,
        alpha=args.point_alpha,
        marker=args.marker,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_title(f"ASTER spatial domains (K={args.k})")
    ax.set_aspect("equal")
    if args.invert_yaxis:
        ax.invert_yaxis()
    ax.axis("off")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(args.vis_dir / f"domain_map_K{args.k}.{ext}", dpi=350, bbox_inches="tight")
    plt.close()

    domain_ids, domain_counts = np.unique(labels, return_counts=True)
    for domain_id, count in zip(domain_ids, domain_counts):
        print(f"  domain {domain_id:2d}: {count:8,d} ({count / len(labels) * 100:5.2f}%)")

    save_json(
        args.vis_dir / "cluster_summary.json",
        {
            "k": args.k,
            "n_cells": int(latent.shape[0]),
            "smooth_k": int(args.smooth_k),
            "kmeans_random_state": int(args.kmeans_random_state),
            "kmeans_n_init": args.kmeans_n_init,
            "palette": args.palette,
            "domain_sizes": {int(d): int(c) for d, c in zip(domain_ids, domain_counts)},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
