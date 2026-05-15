from __future__ import annotations

import argparse
import random
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from repro_st_aster.aster_ntd import ASTER, load_dlpfc_slice
from repro_st_aster.common import ensure_dir, save_json


DEFAULT_SLICES = [
    "151507",
    "151508",
    "151509",
    "151510",
    "151669",
    "151670",
    "151671",
    "151672",
    "151673",
    "151674",
    "151675",
    "151676",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mclust_cluster(x: np.ndarray, n_clusters: int, r_home: str | None = None) -> np.ndarray:
    if r_home:
        import os

        os.environ["R_HOME"] = r_home

    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    robjects.r.library("mclust")
    mclust = robjects.r["Mclust"]
    result = mclust(x, n_clusters)
    labels = np.array(result.rx2("classification")).astype(int)
    return labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASTER on preprocessed DLPFC slices and compute ARI with Mclust.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/dlpfc"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/dlpfc"))
    parser.add_argument("--slice-ids", nargs="*", default=DEFAULT_SLICES)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--rank-x", type=int, default=64)
    parser.add_argument("--rank-y", type=int, default=64)
    parser.add_argument("--rank-g", type=int, default=256)
    parser.add_argument("--mid-feature", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-epoch", type=int, default=1500)
    parser.add_argument("--n-clusters", type=int, default=7)
    parser.add_argument("--n-pca", type=int, default=15)
    parser.add_argument("--r-home", type=str, default=None)
    parser.add_argument("--skip-mclust", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    if args.dry_run:
        for slice_id in args.slice_ids:
            print(args.data_dir / f"{slice_id}_preprocessed.h5ad")
            print(args.data_dir / f"{slice_id}_truth.txt")
        return 0

    set_seed(args.seed)
    all_results = []
    for slice_id in args.slice_ids:
        h5ad_path = args.data_dir / f"{slice_id}_preprocessed.h5ad"
        truth_path = args.data_dir / f"{slice_id}_truth.txt"
        run_dir = ensure_dir(args.out_dir / f"slice_{slice_id}")
        data = load_dlpfc_slice(str(h5ad_path), str(truth_path), use_all_entries=True)
        model = ASTER(
            expr_tensor=data["expr_tensor"],
            gene_names=data["gene_names"],
            n_x=data["n_x"],
            n_y=data["n_y"],
            coords_mapping=data["coords_mapping"],
        )
        model.fit(
            rank_x=args.rank_x,
            rank_y=args.rank_y,
            rank_g=args.rank_g,
            mid_feature=args.mid_feature,
            lr=args.lr,
            max_epoch=args.max_epoch,
            save_dir=str(run_dir),
            verbose=False,
        )

        expr_imputed, gene_names = model.get_imputed_matrix()
        pca = PCA(n_components=args.n_pca, random_state=args.seed)
        expr_pca = pca.fit_transform(expr_imputed)

        ari = None
        pred_labels = None
        if not args.skip_mclust:
            pred_labels = mclust_cluster(expr_pca, args.n_clusters, args.r_home)
            truth_labels = pd.Series(data["labels_gt"]).astype(str).to_numpy()
            valid = ~pd.isna(truth_labels)
            ari = float(adjusted_rand_score(truth_labels[valid], pred_labels[valid]))

        adata = anndata.read_h5ad(h5ad_path)
        adata.layers["ASTER_imputed"] = expr_imputed
        if pred_labels is not None:
            adata.obs["ASTER_cluster"] = pred_labels.astype(str)
        adata.write_h5ad(run_dir / f"{slice_id}_ASTER_imputed.h5ad")

        row = {"Slice_ID": int(slice_id), "Seed": args.seed, "ARI": ari}
        all_results.append(row)
        save_json(run_dir / "summary.json", row)
        model.clear_gpu()

    df = pd.DataFrame(all_results)
    df.to_csv(args.out_dir / "DLPFC_ASTER_ARI_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
