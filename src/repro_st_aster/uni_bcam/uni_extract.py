from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from repro_st_aster.common import ensure_dir, file_report, save_json

# The standardized H&E images are far larger than PIL's decompression-bomb guard
# (the gastric slide is 40096 x 16352), so the limit is lifted deliberately.
Image.MAX_IMAGE_PIXELS = None

TILE_SIZE = 224
PATCHES_PER_TILE = 16
FEATURE_DIM = 1536


def build_model(model_dir: Path, device):
    import torch
    import timm

    kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(pretrained=False, **kwargs)
    weights_path = model_dir / "pytorch_model.bin"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    return model.to(device).eval()


def extract_features(
    image_path: Path,
    model_dir: Path,
    out_dir: Path,
    device: str | None = None,
    superpixel_stride: int = 16,
    write_pickle: bool = True,
    max_tile_rows: Optional[int] = None,
    tile_batch_size: Optional[int] = None,
):
    """Extract a UNI-2 superpixel feature grid from a standardized H&E image.

    The image is cut into 224-px tiles; each tile yields 16x16 patch tokens, which
    are stitched into a ``(n_tile_rows * 16, n_tile_cols * 16, 1536)`` grid.

    ``superpixel_stride`` is recorded in the metadata only -- it does not change the
    grid, it tells the downstream mapping step how many pixels one superpixel spans.
    ``max_tile_rows`` truncates to the first N tile rows (smoke tests), and
    ``tile_batch_size`` splits a tile row into chunks so that wide slides do not
    exhaust GPU memory.
    """
    import torch
    from torchvision import transforms

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ensure_dir(out_dir)

    model = build_model(model_dir, device)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    if w % TILE_SIZE or h % TILE_SIZE:
        raise ValueError(f"image size must be divisible by {TILE_SIZE}: {w}x{h}")

    n_patches_w = w // TILE_SIZE
    n_patches_h = h // TILE_SIZE
    if max_tile_rows is not None:
        n_patches_h = min(n_patches_h, max_tile_rows)
    chunk = tile_batch_size or n_patches_w

    features_rows = []
    patch_cache = []

    def hook_fn(_module, _inp, output):
        # Drop the 1 cls + 8 register tokens, keep the 256 patch tokens.
        patch_cache.append(
            output[:, 9:, :].reshape(output.shape[0], PATCHES_PER_TILE, PATCHES_PER_TILE, FEATURE_DIM).cpu().numpy()
        )

    handle = model.norm.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for row_idx in range(n_patches_h):
                row_blocks = []
                for col_start in range(0, n_patches_w, chunk):
                    tiles = []
                    for col_idx in range(col_start, min(col_start + chunk, n_patches_w)):
                        left = col_idx * TILE_SIZE
                        upper = row_idx * TILE_SIZE
                        tiles.append(transform(image.crop((left, upper, left + TILE_SIZE, upper + TILE_SIZE))))
                    patch_cache.clear()
                    _ = model(torch.stack(tiles).to(device))
                    row_blocks.append(np.concatenate(patch_cache, axis=0))
                features_rows.append(np.concatenate(row_blocks, axis=0))
    finally:
        handle.remove()

    stitched_rows = [np.concatenate(row, axis=1) for row in features_rows]
    feature_map = np.concatenate(stitched_rows, axis=0)

    grid_h, grid_w, feat_dim = feature_map.shape
    np.save(out_dir / "superpixel_features.npy", feature_map)

    i, j = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    coords = {
        "i": i,
        "j": j,
        "x_std": (j + 0.5) * superpixel_stride,
        "y_std": (i + 0.5) * superpixel_stride,
        "x_um": (j + 0.5) * superpixel_stride * 0.5,
        "y_um": (i + 0.5) * superpixel_stride * 0.5,
    }
    np.savez(out_dir / "superpixel_coordinates.npz", **coords)

    metadata = {
        "feature_dim": int(feat_dim),
        "grid_shape": [int(grid_h), int(grid_w)],
        "image_size": [int(w), int(h)],
        "model": "UNI-2 (ViT-Giant/14)",
        "patch_size": TILE_SIZE,
        "superpixel_size_pixels": int(superpixel_stride),
        "superpixel_size_um": float(superpixel_stride) * 0.5,
        "n_tiles": [int(n_patches_h), int(n_patches_w)],
        "truncated_tile_rows": max_tile_rows is not None,
    }
    save_json(out_dir / "extraction_metadata.json", metadata)
    if write_pickle:
        with (out_dir / "uni2_features_complete.pkl").open("wb") as fh:
            pickle.dump({"features": feature_map, "coordinates": coords, "metadata": metadata}, fh)
    return metadata


def main(argv: list[str] | None = None) -> int:
    """Entry point; ``argv`` lets a notebook call this like a function."""
    parser = argparse.ArgumentParser(description="Extract UNI-2 features from a standardized H&E image.")
    parser.add_argument("--image-path", type=Path, default=Path("raw_data/bc_xenium/tissue_standardized_0p5um.jpg"))
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing UNI-2 weights (pytorch_model.bin).")
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/bc_xenium/uni"))
    parser.add_argument(
        "--superpixel-stride",
        type=int,
        default=16,
        help="Pixel step per superpixel recorded in the metadata (colorectal used 14, single-cell runs 16).",
    )
    parser.add_argument("--no-pickle", action="store_true", help="Skip the large uni2_features_complete.pkl duplicate.")
    parser.add_argument("--max-tile-rows", type=int, default=None, help="Only process the first N tile rows (smoke test).")
    parser.add_argument("--tile-batch-size", type=int, default=None, help="Tiles per forward pass; defaults to a whole tile row.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(file_report(args.image_path))
        print(file_report(args.model_dir / "pytorch_model.bin"))
        print(f"output -> {args.out_dir}")
        return 0

    metadata = extract_features(
        args.image_path,
        args.model_dir,
        args.out_dir,
        superpixel_stride=args.superpixel_stride,
        write_pickle=not args.no_pickle,
        max_tile_rows=args.max_tile_rows,
        tile_batch_size=args.tile_batch_size,
    )
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
