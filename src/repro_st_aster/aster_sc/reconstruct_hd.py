"""ASTER-SC INR stage for Visium HD (colorectal cancer, Fig. 4).

Kept separate from ``reconstruct.py`` because the training objective differs rather
than just the hyperparameters: this variant trains on all spots with no validation
split, balances the zero/non-zero ratio within each batch, and optimises a weighted
MSE instead of a weighted Huber loss. The published reconstruction is taken from a
periodic snapshot (epoch 4000 of an 8000-epoch schedule) rather than from a
best-validation checkpoint, so ``--recon-epoch`` selects which snapshot to emit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from repro_st_aster.common import ensure_dir, require_inputs, save_json, seed_everything

R_S, R_G, HIDDEN = 512, 256, 512


def main(argv: list[str] | None = None) -> int:
    """Entry point; ``argv`` lets a notebook call this like a function."""
    parser = argparse.ArgumentParser(description="ASTER-SC INR + Tucker-2 on Visium HD bins.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/crc_visiumhd/bcam_input"))
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/crc_visiumhd/inr_output"))
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--batch-spots", type=int, default=32768)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--omega-start", type=float, default=1.0)
    parser.add_argument("--omega-end", type=float, default=10.0)
    parser.add_argument("--omega-ramp", type=float, nargs=2, default=(0.10, 0.70))
    parser.add_argument("--xy-jitter", type=float, default=0.003)
    parser.add_argument("--target-nz-ratio", type=float, default=0.30, help="Non-zero fraction targeted by balanced sampling.")
    parser.add_argument("--nz-weight", type=float, default=3.0, help="Loss weight applied to non-zero entries.")
    parser.add_argument("--snapshot-every", type=int, default=500)
    parser.add_argument(
        "--recon-epoch",
        type=int,
        default=4000,
        help="Emit the reconstruction from this epoch's snapshot and stop (the published run used 4000).",
    )
    parser.add_argument(
        "--from-snapshot",
        type=Path,
        default=None,
        help="Skip training: load these weights and only write the reconstruction.",
    )
    parser.add_argument(
        "--expr-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Where to hold the expression matrix; 'cpu' trades speed for GPU memory.",
    )
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print(args.data_dir / "gene_expression_normalized.npy")
        print(args.data_dir / "cell_coords_standardized.npy")
        print(args.data_dir / "gene_names.npy")
        print(f"output -> {args.out_dir}")
        return 0

    import torch
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast

    from repro_st_aster.aster_sc.inr_model import LRT_Tucker2, omega_ramp

    require_inputs(
        [args.data_dir / "gene_expression_normalized.npy", args.data_dir / "cell_coords_standardized.npy"],
        "crc_visiumhd",
    )

    seed_everything(args.seed)
    ensure_dir(args.out_dir)
    snapshot_dir = ensure_dir(args.out_dir / "snapshots")

    gene_expr = np.load(args.data_dir / "gene_expression_normalized.npy")
    coords = np.load(args.data_dir / "cell_coords_standardized.npy")
    if args.max_cells is not None:
        keep = min(args.max_cells, gene_expr.shape[0])
        gene_expr = gene_expr[:keep]
        coords = coords[:keep]
    n_spot, n_gene = gene_expr.shape

    xy01 = ((coords - coords.min(0)) / np.maximum(np.ptp(coords, 0), 1e-8)).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expr_device = device if (args.expr_device == "cuda" and device.type == "cuda") else torch.device("cpu")
    xy01_gpu = torch.tensor(xy01, device=device)
    expr_all = torch.tensor(gene_expr, device=expr_device)
    all_gene_idx = torch.arange(n_gene, device=device, dtype=torch.long)

    # CRC used the plain xavier_uniform_ default for the coupling matrix.
    model = LRT_Tucker2(
        n_gene,
        r_s=R_S,
        r_g=R_G,
        hidden=HIDDEN,
        depth=args.depth,
        omega_0=args.omega_start,
        p_dropout=args.dropout,
        k_init_gain=None,
    ).to(device)

    def write_reconstruction(tag: str):
        model.eval()
        recon = np.zeros((n_spot, n_gene), dtype=np.float32)
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            for start in range(0, n_spot, 4096):
                block = model.full_reconstruct(xy01_gpu[start : start + 4096])
                recon[start : start + 4096] = F.relu(block).float().cpu().numpy()
        np.save(args.out_dir / "inr_reconstructed_expression.npy", recon)
        np.save(args.out_dir / "coords_original.npy", coords)
        np.save(args.out_dir / "coords_01.npy", xy01)
        print(f"reconstruction ({tag}): {recon.shape}, max={recon.max():.4f}")
        return recon

    if args.from_snapshot is not None:
        model.load_state_dict(torch.load(args.from_snapshot, map_location=device), strict=True)
        write_reconstruction(f"snapshot {args.from_snapshot.name}")
        save_json(
            args.out_dir / "inr_training_info.json",
            {
                "n_cells": int(n_spot),
                "n_genes": int(n_gene),
                "depth": int(args.depth),
                "from_snapshot": str(args.from_snapshot),
                "epochs_run": 0,
            },
        )
        return 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")

    def lr_at(epoch: int) -> float:
        if epoch < args.warmup:
            return args.lr * (epoch + 1) / args.warmup
        t = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * t))

    idx_train = np.arange(n_spot)  # all spots; no validation split
    log_rows = []
    epochs_run = 0
    recon_epoch = min(args.recon_epoch, args.epochs) if args.recon_epoch > 0 else args.epochs

    for epoch in range(1, args.epochs + 1):
        model.train()
        omega = omega_ramp(epoch, args.epochs, args.omega_start, args.omega_end, tuple(args.omega_ramp))
        model.set_omega(omega)
        cur_lr = lr_at(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr
        np.random.shuffle(idx_train)

        losses, nz_ratios = [], []
        for start in range(0, n_spot, args.batch_spots):
            batch_idx = torch.tensor(idx_train[start : start + args.batch_spots], device=device, dtype=torch.long)
            if len(batch_idx) == 0:
                continue
            xy_b = xy01_gpu[batch_idx]
            if args.xy_jitter > 0:
                xy_b = torch.clamp(xy_b + torch.randn_like(xy_b) * args.xy_jitter, 0.0, 1.0)
            y = expr_all[batch_idx.to(expr_device)].to(device, non_blocking=True)

            with torch.no_grad():
                # Balanced sampling: keep every non-zero, subsample zeros so that
                # non-zeros make up target_nz_ratio of the supervised entries.
                mask_nz = y > 0
                n_nonzero = int(mask_nz.sum())
                if n_nonzero == 0:
                    continue
                mask_zero = y == 0
                n_zero_available = int(mask_zero.sum())
                n_zero_needed = int(n_nonzero / args.target_nz_ratio) - n_nonzero
                if n_zero_available > 0 and n_zero_needed > 0:
                    keep_ratio = min(1.0, n_zero_needed / n_zero_available)
                    mask = mask_nz | (mask_zero & (torch.rand_like(y.float()) < keep_ratio))
                else:
                    mask = mask_nz
                nz_ratios.append(float(mask_nz[mask].float().mean()))

            with autocast(enabled=device.type == "cuda"):
                pred = F.relu(model.forward_block(xy_b, all_gene_idx))
                weight = torch.where(y > 0, args.nz_weight, 1.0)
                diff = pred - y
                loss = (weight * diff * diff * mask).sum() / (weight * mask).sum()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        epochs_run = epoch
        train_loss = float(np.mean(losses)) if losses else float("nan")
        nz_ratio = float(np.mean(nz_ratios)) if nz_ratios else float("nan")
        log_rows.append({"epoch": epoch, "lr": cur_lr, "omega": omega, "train_loss": train_loss, "actual_nz_ratio": nz_ratio})
        if epoch % 50 == 0 or epoch == 1:
            print(f"  INR ep {epoch:5d} | omega={omega:.2f} | lr={cur_lr:.2e} | loss={train_loss:.4f} | nz={nz_ratio:.2%}")

        if args.snapshot_every > 0 and epoch % args.snapshot_every == 0:
            torch.save(model.state_dict(), snapshot_dir / f"model_epoch_{epoch:05d}.pt")
            print(f"  snapshot saved: epoch {epoch}")

        if epoch == recon_epoch:
            torch.save(model.state_dict(), snapshot_dir / f"model_epoch_{epoch:05d}.pt")
            print(f"  reached recon epoch {epoch}; emitting reconstruction and stopping")
            break

    pd.DataFrame(log_rows).to_csv(args.out_dir / "train_log.csv", index=False)
    write_reconstruction(f"epoch {epochs_run}")
    torch.save(model.state_dict(), args.out_dir / "inr_tucker_model.pth")
    save_json(
        args.out_dir / "inr_training_info.json",
        {
            "n_cells": int(n_spot),
            "n_genes": int(n_gene),
            "depth": int(args.depth),
            "epochs_run": int(epochs_run),
            "epochs_requested": int(args.epochs),
            "recon_epoch": int(recon_epoch),
            "final_train_loss": float(log_rows[-1]["train_loss"]) if log_rows else None,
            "target_nz_ratio": float(args.target_nz_ratio),
            "nz_weight": float(args.nz_weight),
            "xy_jitter": float(args.xy_jitter),
            "loss": "balanced weighted MSE",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
