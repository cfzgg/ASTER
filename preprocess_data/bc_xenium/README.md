# `preprocess_data/bc_xenium/` — included in the archive or generated

> The listed files are included in the single archive linked in the root README. This
> file documents the expected filenames and stage outputs; no separate download is needed.

All subdirectories ship **empty** except for a `.gitkeep`.

`notebooks/BC_Xenium_Breast_Cancer_Demo.ipynb` reviews **precomputed** outputs, so to
run that notebook as written you need the download. To regenerate everything from
`raw_data/bc_xenium/` instead, run the five `scripts/*_bc_xenium_*.sh` in order and
only `uni/` needs to come from a download (or your own UNI-2 weights).

| Subdirectory  | Filled by | Contents |
|---------------|-----------|----------|
| `uni/`        | `prepare_bc_xenium_uni.sh`, **or** the root archive | `superpixel_features.npy` `(496, 672, 1536)` 1.9 GB, `superpixel_coordinates.npz`, `extraction_metadata.json` |
| `bcam_input/` | `prepare_bc_xenium_bcam_input.sh`, **or** the root archive | `gene_expression_normalized.npy` `(167780, 313)`, `uni2_features_per_cell.npy` `(167780, 1536)`, `cell_coords_standardized.npy`, `gene_names.npy`, `cell_ids.npy`, `superpixel_mapping.npy`, `preprocessing_metadata.json` |
| `inr_output/` | `reproduce_bc_xenium_inr.sh`, **or** the root archive | `inr_reconstructed_expression.npy` `(167780, 313)`, `inr_tucker_model.pth`, `coords_01.npy`, `coords_original.npy`, `training_curves.png`, `inr_training_info.json` |
| `bcam_output/`| `reproduce_bc_xenium_bcam.sh`, **or** the root archive | `fusion_latent_512.npy` `(167780, 512)`, `fusion_predicted_expression.npy`, `bcam_best.pth`, `bcam_final.pth`, `training_info.json` |
| `viz/`        | `reproduce_bc_xenium_cluster.sh`, **or** the root archive | `labels_fusion_K17.npy`, `domain_map_K17.{png,pdf}`, `clustering_K17_overview.png`, `cell_clusters_K17.csv`, `cluster_summary.json` |

Roughly 3.9 GB in total.
