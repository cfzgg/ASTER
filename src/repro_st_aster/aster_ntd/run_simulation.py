from __future__ import annotations

import argparse
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from repro_st_aster.aster_ntd import ASTER, load_simulation_slice
from repro_st_aster.common import ensure_dir, save_json, seed_everything


DEFAULT_SCENARIOS = [
    "sr_norm_0.3",
    "sr_norm_0.4",
    "sr_norm_0.5",
    "noise_norm_sd_0.5",
    "noise_norm_sd_1.0",
    "noise_norm_sd_2.0",
]
DEFAULT_SECTIONS = ["embryo_section1", "embryo_section2", "embryo_section3"]


def compute_metrics(adata, imputed_layer: str, viz_genes: list[str]) -> dict:
    y_true = adata.layers["original_norm"]
    y_pred = adata.layers[imputed_layer]
    if sp.issparse(y_true):
        y_true = y_true.toarray()
    if sp.issparse(y_pred):
        y_pred = y_pred.toarray()

    tmp = adata.copy()
    tmp.X = tmp.layers["original_norm"]
    sc.pp.highly_variable_genes(tmp, n_top_genes=100, flavor="seurat_v3")
    hvg_idx = np.where(tmp.var["highly_variable"].values)[0]

    mse_hvg = float(np.mean((y_true[:, hvg_idx] - y_pred[:, hvg_idx]) ** 2))
    rmse_hvg = float(np.sqrt(mse_hvg))
    nrmse = rmse_hvg / float(np.std(y_true[:, hvg_idx]))
    mse_all = float(np.mean((y_true - y_pred) ** 2))

    gene_metrics = {}
    for gene in viz_genes:
        matches = [g for g in adata.var_names if g.lower() == gene.lower()]
        if not matches:
            continue
        gname = matches[0]
        gi = adata.var_names.get_loc(gname)
        if not isinstance(gi, (int, np.integer)):
            gi = gi[0]
        t, p = y_true[:, gi], y_pred[:, gi]
        corr = 0.0 if t.std() == 0 or p.std() == 0 else float(np.corrcoef(t, p)[0, 1])
        gene_metrics[gname] = {"mse": float(np.mean((t - p) ** 2)), "corr": corr}
    return {"mse_hvg": mse_hvg, "rmse_hvg": rmse_hvg, "nrmse": nrmse, "mse_all": mse_all, "gene_metrics": gene_metrics}


def visualize(adata, gene: str, input_layer: str, imputed_layer: str, save_path: Path) -> None:
    matches = [g for g in adata.var_names if g.lower() == gene.lower()]
    if not matches:
        return
    gname = matches[0]
    gi = adata.var_names.get_loc(gname)
    if not isinstance(gi, (int, np.integer)):
        gi = gi[0]

    xc = adata.obs["array_col"].values
    yc = adata.obs["array_row"].values

    def dense(layer_key):
        d = adata.layers[layer_key]
        return d.toarray() if sp.issparse(d) else d

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, lkey) in zip(
        axes,
        [("Original", "original_norm"), (f"Input ({input_layer})", input_layer), ("ASTER", imputed_layer)],
    ):
        ax.scatter(xc, yc, c=dense(lkey)[:, gi], cmap="RdYlBu_r", s=5)
        ax.set_title(title, fontsize=14)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.axis("off")
    plt.suptitle(f"Gene: {gname}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASTER simulation benchmark on preprocessed embryo sections.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/simulation"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/simulation"))
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-epoch", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sections", nargs="*", default=DEFAULT_SECTIONS)
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--viz-genes", nargs="*", default=["Gm42418", "Rpl41"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.out_dir)
    if args.dry_run:
        for section in args.sections:
            print(args.data_dir / f"{section}_processed.h5ad")
        return 0

    all_metrics = []
    for section in args.sections:
        h5ad_path = args.data_dir / f"{section}_processed.h5ad"
        if not h5ad_path.exists():
            print(f"Missing: {h5ad_path}")
            continue
        adata_full = sc.read_h5ad(h5ad_path)

        for scenario in args.scenarios:
            run_dir = ensure_dir(args.out_dir / section / scenario)
            data = load_simulation_slice(str(h5ad_path), scenario)
            model = ASTER(
                expr_tensor=data["expr_tensor"],
                gene_names=data["gene_names"],
                n_x=data["n_x"],
                n_y=data["n_y"],
                coords_mapping=data["coords_mapping"],
            )
            model.fit(rank_x=args.rank, rank_y=args.rank, rank_g=args.rank, lr=args.lr, max_epoch=args.max_epoch, save_dir=str(run_dir), verbose=False)
            expr_imputed, _ = model.get_imputed_matrix()
            imputed_key = f"ASTER_{scenario}"
            adata_full.layers[imputed_key] = sp.csr_matrix(expr_imputed)

            metrics = compute_metrics(adata_full, imputed_key, args.viz_genes)
            row = {"section": section, "scenario": scenario, **{k: v for k, v in metrics.items() if k != "gene_metrics"}}
            all_metrics.append(row)

            out_h5ad = run_dir / f"{section}_{scenario}_imputed.h5ad"
            adata_out = anndata.AnnData(X=adata_full.X.copy(), obs=adata_full.obs.copy(), var=adata_full.var.copy(), uns=adata_full.uns.copy())
            adata_out.layers["original_norm"] = adata_full.layers["original_norm"]
            adata_out.layers[scenario] = adata_full.layers[scenario]
            adata_out.layers[imputed_key] = adata_full.layers[imputed_key]
            adata_out.write_h5ad(out_h5ad)
            for gene in args.viz_genes:
                visualize(adata_full, gene, scenario, imputed_key, run_dir / f"{gene}_{scenario}.png")
            save_json(run_dir / "metrics.json", metrics)
            model.clear_gpu()

    df = pd.DataFrame(all_metrics)
    df.to_csv(args.out_dir / "simulation_metrics_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
