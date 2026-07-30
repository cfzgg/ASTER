#!/usr/bin/env python3
"""Check that a downloaded dataset is unpacked where the workflows expect it.

This repository ships every data directory empty, so a wrong unpack path is the most
likely first failure. Run this before a workflow to get a clear report instead of a
FileNotFoundError several minutes in.

Usage:
    python scripts/check_data.py                 # all datasets
    python scripts/check_data.py bc_xenium       # one dataset
    python scripts/check_data.py crc_visiumhd gastric_xenium

Exit status is 0 only if every required file of every requested dataset is present.
Optional files (needed only for a specific route) are reported but never fail the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (relative path, required?, note)
DATASETS: dict[str, list[tuple[str, bool, str]]] = {
    "bc_xenium": [
        ("raw_data/bc_xenium/cell_feature_matrix.h5", True, ""),
        ("raw_data/bc_xenium/cell_coordinates.csv", True, ""),
        ("raw_data/bc_xenium/tissue_standardized_0p5um.jpg", True, ""),
        ("preprocess_data/bc_xenium/uni/superpixel_features.npy", False, "UNI-2 route B; or extract with your own weights"),
        ("preprocess_data/bc_xenium/bcam_input/gene_expression_normalized.npy", False, "precomputed; or run prepare_bc_xenium_bcam_input.sh"),
        ("preprocess_data/bc_xenium/inr_output/inr_reconstructed_expression.npy", False, "precomputed; or run reproduce_bc_xenium_inr.sh"),
        ("preprocess_data/bc_xenium/bcam_output/fusion_latent_512.npy", False, "precomputed; or run reproduce_bc_xenium_bcam.sh"),
        ("preprocess_data/bc_xenium/viz/labels_fusion_K17.npy", False, "precomputed; or run reproduce_bc_xenium_cluster.sh"),
    ],
    "crc_visiumhd": [
        ("raw_data/crc_visiumhd/binned_outputs/square_008um/filtered_feature_bc_matrix/barcodes.tsv.gz", True, ""),
        ("raw_data/crc_visiumhd/binned_outputs/square_008um/filtered_feature_bc_matrix/features.tsv.gz", True, ""),
        ("raw_data/crc_visiumhd/binned_outputs/square_008um/filtered_feature_bc_matrix/matrix.mtx.gz", True, ""),
        ("raw_data/crc_visiumhd/binned_outputs/square_008um/spatial/tissue_positions.parquet", True, ""),
        ("raw_data/crc_visiumhd/metadata.json", True, "H&E crop bbox + scale factor"),
        ("raw_data/crc_visiumhd/tissue_standardized_0p5um.jpg", False, "only for UNI-2 route A"),
        ("preprocess_data/crc_visiumhd/uni/superpixel_features.npy", False, "UNI-2 route B; or extract with your own weights"),
    ],
    "gastric_xenium": [
        ("raw_data/gastric_xenium/cell_feature_matrix.h5", True, ""),
        ("raw_data/gastric_xenium/cell_coordinates.csv", True, ""),
        ("raw_data/gastric_xenium/tissue_standardized_0p5um.jpg", False, "only for UNI-2 route A"),
        ("preprocess_data/gastric_xenium/uni/superpixel_features.npy", False, "UNI-2 route B; or extract with your own weights"),
    ],
    "dlpfc": [
        (f"preprocess_data/dlpfc/{sid}_preprocessed.h5ad", True, "")
        for sid in (151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676)
    ]
    + [
        (f"preprocess_data/dlpfc/{sid}_truth.txt", False, "only for the ARI evaluation")
        for sid in (151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676)
    ],
    "simulation": [
        (f"preprocess_data/simulation/embryo_section{i}_processed.h5ad", True, "") for i in (1, 2, 3)
    ],
}


def human(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def check(name: str) -> bool:
    entries = DATASETS[name]
    missing_required, missing_optional, present = [], [], []
    for rel, required, note in entries:
        path = REPO_ROOT / rel
        if path.exists():
            present.append((rel, path.stat().st_size))
        elif required:
            missing_required.append((rel, note))
        else:
            missing_optional.append((rel, note))

    total = sum(size for _, size in present)
    status = "OK" if not missing_required else "INCOMPLETE"
    print(f"\n=== {name}: {status} ===")
    print(f"  present: {len(present)}/{len(entries)} files, {human(total)}")

    if missing_required:
        print(f"  MISSING (required, {len(missing_required)}):")
        for rel, note in missing_required:
            print(f"    - {rel}" + (f"   [{note}]" if note else ""))
    if missing_optional:
        print(f"  missing (optional, {len(missing_optional)}):")
        shown = missing_optional[:6]
        for rel, note in shown:
            print(f"    - {rel}" + (f"   [{note}]" if note else ""))
        if len(missing_optional) > len(shown):
            print(f"    ... and {len(missing_optional) - len(shown)} more")
    if missing_required:
        readme = REPO_ROOT / Path(missing_required[0][0]).parts[0] / name / "README.md"
        hint = readme if readme.exists() else REPO_ROOT / "README.md"
        print(f"  -> see {hint.relative_to(REPO_ROOT)} for the download link and layout")
    return not missing_required


def main(argv: list[str]) -> int:
    names = argv or sorted(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        print(f"unknown dataset(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(sorted(DATASETS))}", file=sys.stderr)
        return 2
    results = {name: check(name) for name in names}
    ok = sum(results.values())
    print(f"\n{ok}/{len(results)} dataset(s) ready" + ("" if ok == len(results) else " -- see MISSING entries above"))
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
