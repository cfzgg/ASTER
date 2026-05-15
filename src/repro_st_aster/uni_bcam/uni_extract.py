from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from repro_st_aster.common import ensure_dir, file_report, save_json


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
):
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
    if w % 224 or h % 224:
        raise ValueError(f"image size must be divisible by 224: {w}x{h}")

    n_patches_w = w // 224
    n_patches_h = h // 224
    features_rows = []
    patch_cache = []

    def hook_fn(_module, _inp, output):
        patch_cache.append(output[:, 9:, :].reshape(output.shape[0], 16, 16, 1536).cpu().numpy())

    handle = model.norm.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            for row_idx in range(n_patches_h):
                row_patches = []
                for col_idx in range(n_patches_w):
                    left = col_idx * 224
                    upper = row_idx * 224
                    patch = image.crop((left, upper, left + 224, upper + 224))
                    row_patches.append(transform(patch))
                batch = torch.stack(row_patches).to(device)
                patch_cache.clear()
                _ = model(batch)
                features_rows.append(patch_cache[0])
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
        "x_std": (j + 0.5) * 16,
        "y_std": (i + 0.5) * 16,
        "x_um": (j + 0.5) * 8.0,
        "y_um": (i + 0.5) * 8.0,
    }
    np.savez(out_dir / "superpixel_coordinates.npz", **coords)

    metadata = {
        "feature_dim": int(feat_dim),
        "grid_shape": [int(grid_h), int(grid_w)],
        "image_size": [int(w), int(h)],
        "model": "UNI-2 (ViT-Giant/14)",
        "patch_size": 224,
        "superpixel_size_pixels": 16,
        "superpixel_size_um": 8.0,
    }
    save_json(out_dir / "extraction_metadata.json", metadata)
    with (out_dir / "uni2_features_complete.pkl").open("wb") as fh:
        pickle.dump({"features": feature_map, "coordinates": coords, "metadata": metadata}, fh)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract UNI-2 features from breast cancer Xenium H&E image.")
    parser.add_argument("--image-path", type=Path, default=Path("raw_data/bc_xenium/tissue_standardized_0p5um.jpg"))
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing UNI-2 weights (pytorch_model.bin).")
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/bc_xenium/uni"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(file_report(args.image_path))
        print(file_report(args.model_dir / "pytorch_model.bin"))
        print(f"output -> {args.out_dir}")
        return 0

    metadata = extract_features(args.image_path, args.model_dir, args.out_dir)
    print(metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
