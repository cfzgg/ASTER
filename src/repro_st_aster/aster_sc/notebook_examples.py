from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from sklearn.decomposition import PCA

from repro_st_aster.common import ensure_dir, save_json


def load_marker_genes(gene_names: np.ndarray, requested: list[str]) -> list[str]:
    upper_to_gene = {str(g).upper(): str(g) for g in gene_names}
    found = [upper_to_gene[g.upper()] for g in requested if g.upper() in upper_to_gene]
    if found:
        return found[:6]
    return [str(g) for g in gene_names[:6]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate figure panels used in the breast cancer visualization notebook.")
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_data/bc_xenium"))
    parser.add_argument("--preprocess-dir", type=Path, default=Path("preprocess_data/bc_xenium"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/BC_Xenium_notebook_examples"))
    parser.add_argument("--pca-sample", type=int, default=5000)
    parser.add_argument("--marker-genes", nargs="*", default=["KRT14", "EPCAM", "CD68", "COL1A1", "PECAM1", "PTPRC"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    uni_dir = args.preprocess_dir / "uni"
    bcam_dir = args.preprocess_dir / "bcam_input"
    inr_dir = args.preprocess_dir / "inr_output"
    bcam_out_dir = args.preprocess_dir / "bcam_output"
    vis_dir = args.preprocess_dir / "viz"

    required = [
        args.raw_dir / "cell_feature_matrix.h5",
        args.raw_dir / "cell_coordinates.csv",
        args.raw_dir / "tissue_standardized_0p5um.jpg",
        uni_dir / "superpixel_features.npy",
        bcam_dir / "gene_names.npy",
        bcam_dir / "cell_coords_standardized.npy",
        inr_dir / "inr_reconstructed_expression.npy",
        bcam_out_dir / "fusion_latent_512.npy",
        bcam_out_dir / "fusion_predicted_expression.npy",
        vis_dir / "labels_fusion_K17.npy",
    ]
    if args.dry_run:
        for path in required:
            print(path)
        return 0

    ensure_dir(args.out_dir)
    adata_raw = sc.read_10x_h5(args.raw_dir / "cell_feature_matrix.h5")
    cell_coords = pd.read_csv(args.raw_dir / "cell_coordinates.csv")
    he_img = Image.open(args.raw_dir / "tissue_standardized_0p5um.jpg")
    uni_features = np.load(uni_dir / "superpixel_features.npy", mmap_mode="r")
    gene_names = np.load(bcam_dir / "gene_names.npy", allow_pickle=True)
    coords = np.load(bcam_dir / "cell_coords_standardized.npy")
    inr_expr = np.load(inr_dir / "inr_reconstructed_expression.npy", mmap_mode="r")
    fusion_latent = np.load(bcam_out_dir / "fusion_latent_512.npy", mmap_mode="r")
    fusion_expr = np.load(bcam_out_dir / "fusion_predicted_expression.npy", mmap_mode="r")
    labels = np.load(vis_dir / "labels_fusion_K17.npy")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(he_img)
    axes[0].set_title("Standardized H&E image")
    axes[0].axis("off")
    axes[1].scatter(cell_coords["x_standardized"], cell_coords["y_standardized"], s=0.2, alpha=0.4)
    axes[1].set_title("Cell coordinates in standardized space")
    axes[1].set_aspect("equal")
    axes[1].invert_yaxis()
    plt.tight_layout()
    plt.savefig(args.out_dir / "01_raw_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    mean_activation = np.asarray(uni_features.mean(axis=2))
    std_activation = np.asarray(uni_features.std(axis=2))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im0 = axes[0].imshow(mean_activation, cmap="viridis", aspect="auto")
    axes[0].set_title("UNI mean activation")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(std_activation, cmap="plasma", aspect="auto")
    axes[1].set_title("UNI feature std")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(args.out_dir / "02_uni_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    n_sample = min(args.pca_sample, fusion_latent.shape[0])
    pca_idx = np.linspace(0, fusion_latent.shape[0] - 1, n_sample, dtype=int)
    latent_sample = np.asarray(fusion_latent[pca_idx])
    latent_pca = PCA(n_components=2, random_state=42).fit_transform(latent_sample)
    label_sample = labels[pca_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(coords[:, 0], coords[:, 1], c=labels, s=0.4, cmap="tab20", alpha=0.8)
    axes[0].set_title("ASTER-SC / BCAM clustering")
    axes[0].set_aspect("equal")
    axes[0].axis("off")
    axes[1].scatter(latent_pca[:, 0], latent_pca[:, 1], c=label_sample, s=2, cmap="tab20", alpha=0.5)
    axes[1].set_title(f"Fusion latent PCA (n={n_sample})")
    cluster_ids, cluster_counts = np.unique(labels, return_counts=True)
    axes[2].bar(cluster_ids, cluster_counts, color=plt.cm.tab20(cluster_ids / max(cluster_ids.max(), 1)))
    axes[2].set_title("Cluster sizes")
    axes[2].set_xlabel("Cluster")
    axes[2].set_ylabel("Cells")
    plt.tight_layout()
    plt.savefig(args.out_dir / "03_aster_sc_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    markers = load_marker_genes(gene_names, args.marker_genes)
    fig, axes = plt.subplots(len(markers), 2, figsize=(10, 4 * len(markers)))
    if len(markers) == 1:
        axes = np.array([axes])
    for i, gene in enumerate(markers):
        gi = np.where(gene_names == gene)[0][0]
        axes[i, 0].scatter(coords[:, 0], coords[:, 1], c=np.asarray(inr_expr[:, gi]), s=0.3, cmap="turbo", alpha=0.8)
        axes[i, 0].set_title(f"{gene} - INR")
        axes[i, 0].set_aspect("equal")
        axes[i, 0].axis("off")
        axes[i, 1].scatter(coords[:, 0], coords[:, 1], c=np.asarray(fusion_expr[:, gi]), s=0.3, cmap="turbo", alpha=0.8)
        axes[i, 1].set_title(f"{gene} - BCAM fusion")
        axes[i, 1].set_aspect("equal")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig(args.out_dir / "04_marker_gene_panels.png", dpi=150, bbox_inches="tight")
    plt.close()

    cluster_mean = []
    for cid in np.unique(labels):
        mask = labels == cid
        cluster_mean.append(np.asarray(fusion_expr[mask]).mean(axis=0))
    cluster_mean = np.asarray(cluster_mean)
    var_idx = np.argsort(cluster_mean.var(axis=0))[-30:]
    heat = cluster_mean[:, var_idx]
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heat.T, aspect="auto", cmap="RdBu_r")
    ax.set_title("Top variable genes across clusters")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Gene")
    ax.set_xticks(range(cluster_mean.shape[0]))
    ax.set_yticks(range(len(var_idx)))
    ax.set_yticklabels(gene_names[var_idx], fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(args.out_dir / "05_cluster_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    save_json(
        args.out_dir / "example_summary.json",
        {
            "raw_shape": list(adata_raw.shape),
            "uni_shape": list(uni_features.shape),
            "inr_shape": list(inr_expr.shape),
            "fusion_latent_shape": list(fusion_latent.shape),
            "fusion_expr_shape": list(fusion_expr.shape),
            "n_clusters": int(len(np.unique(labels))),
            "marker_genes": markers,
            "pca_sample": int(n_sample),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
