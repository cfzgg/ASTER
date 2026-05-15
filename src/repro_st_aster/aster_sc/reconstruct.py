from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from repro_st_aster.common import ensure_dir, save_json, seed_everything


class Timer:
    def __init__(self):
        self.times = {}
        self.start_time = None
        self.current_step = None

    def start_total(self):
        self.start_time = time.time()
        print("=" * 70)
        print("ASTER-SC INR + Tucker-2")
        print(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

    def start_step(self, step_name):
        self.current_step = step_name
        self.times[step_name] = {"start": time.time(), "end": None, "duration": None}

    def end_step(self, step_name=None):
        step_name = step_name or self.current_step
        item = self.times[step_name]
        item["end"] = time.time()
        item["duration"] = item["end"] - item["start"]
        print(f"{step_name}: {self.format_time(item['duration'])}")

    @staticmethod
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.2f}s"
        if seconds < 3600:
            return f"{seconds / 60:.2f}min"
        return f"{seconds / 3600:.2f}h"


def main() -> int:
    parser = argparse.ArgumentParser(description="ASTER-SC step 1: INR + Tucker-2 reconstruction.")
    parser.add_argument("--data-dir", type=Path, default=Path("preprocess_data/bc_xenium/bcam_input"))
    parser.add_argument("--out-dir", type=Path, default=Path("preprocess_data/bc_xenium/inr_output"))
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    if args.dry_run:
        print(args.data_dir / "gene_expression_normalized.npy")
        print(args.data_dir / "cell_coords_standardized.npy")
        print(args.data_dir / "gene_names.npy")
        print(f"output -> {args.out_dir}")
        return 0

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import GradScaler, autocast

    timer = Timer()
    timer.start_total()
    ensure_dir(args.out_dir)

    gene_expr = np.load(args.data_dir / "gene_expression_normalized.npy")
    coords = np.load(args.data_dir / "cell_coords_standardized.npy")
    if args.max_cells is not None:
        keep = min(args.max_cells, gene_expr.shape[0])
        gene_expr = gene_expr[:keep]
        coords = coords[:keep]
    n_cells, n_genes = gene_expr.shape

    xy_min = coords.min(axis=0)
    xy_ptp = coords.ptp(axis=0)
    coords_01 = np.clip((coords - xy_min) / xy_ptp, 0.0, 1.0).astype(np.float32)

    class SineLayer(nn.Module):
        def __init__(self, in_features, out_features, omega_0=1.0):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            self.register_buffer("omega", torch.tensor(float(omega_0)))
            with torch.no_grad():
                bound = np.sqrt(6 / in_features) / self.omega.item()
                self.linear.weight.uniform_(-bound, bound)

        def set_omega(self, new_omega: float):
            self.omega.fill_(float(new_omega))

        def forward(self, x):
            return torch.sin(self.omega * self.linear(x))

    class XYNetSIREN(nn.Module):
        def __init__(self, hidden=512, depth=6, out_dim=512, omega_0=1.0, p_dropout=0.1):
            super().__init__()
            layers = [SineLayer(2, hidden, omega_0)]
            for _ in range(depth - 2):
                layers.append(SineLayer(hidden, hidden, omega_0))
                if p_dropout > 0:
                    layers.append(nn.Dropout(p_dropout))
            layers.append(nn.Linear(hidden, out_dim))
            self.net = nn.Sequential(*layers)

        def set_omega(self, omega: float):
            for module in self.net:
                if isinstance(module, SineLayer):
                    module.set_omega(omega)

        def forward(self, xy01):
            return self.net(xy01)

    class LRT_Tucker2(nn.Module):
        def __init__(self, n_genes, r_s=512, r_g=256, hidden=512, depth=6, omega_0=1.0, p_dropout=0.1):
            super().__init__()
            self.xy_net = XYNetSIREN(hidden=hidden, depth=depth, out_dim=r_s, omega_0=omega_0, p_dropout=p_dropout)
            self.g_emb = nn.Embedding(n_genes, r_g)
            self.K = nn.Parameter(torch.empty(r_s, r_g))
            nn.init.xavier_uniform_(self.K, gain=0.5)
            nn.init.normal_(self.g_emb.weight, std=0.02)

        def set_omega(self, omega: float):
            self.xy_net.set_omega(omega)

        def forward_block(self, xy_block, gene_idx_block):
            s = self.xy_net(xy_block)
            g = self.g_emb(gene_idx_block)
            return torch.matmul(torch.matmul(s, self.K), g.t())

        @torch.no_grad()
        def full_reconstruct(self, xy_all):
            s = self.xy_net(xy_all)
            return torch.matmul(torch.matmul(s, self.K), self.g_emb.weight.t())

    model = LRT_Tucker2(n_genes=n_genes)
    if torch.cuda.is_available():
        model = model.cuda()
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scaler = GradScaler(enabled=True)

    idx = np.arange(n_cells)
    np.random.shuffle(idx)
    n_val = max(1, int(n_cells * 0.05))
    idx_val, idx_train = idx[:n_val], idx[n_val:]

    coords_val_gpu = torch.tensor(coords_01[idx_val], dtype=torch.float32, device=device)
    expr_val = gene_expr[idx_val]

    def omega_schedule(epoch, max_epoch):
        a, b = 0.2, 0.6
        p = epoch / max_epoch
        if p <= a:
            return 1.0
        if p >= b:
            return 5.0
        t = (p - a) / (b - a)
        return 1.0 + t * 4.0

    def lr_schedule(base_lr, epoch, max_epoch):
        warmup = 20
        if epoch < warmup:
            return base_lr * (epoch + 1) / warmup
        t = (epoch - warmup) / max(1, max_epoch - warmup)
        return base_lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t)))

    def weighted_huber_loss(pred, target, zero_weight=0.1):
        weight = torch.ones_like(target)
        weight[target == 0] = zero_weight
        return (weight * F.smooth_l1_loss(pred, target, reduction="none")).mean()

    timer.start_step("training")
    best_state = None
    best_val = float("inf")
    train_losses, val_losses = [], []
    gene_idx = torch.arange(n_genes, dtype=torch.long, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.set_omega(omega_schedule(epoch, args.epochs))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_schedule(5e-4, epoch, args.epochs)
        np.random.shuffle(idx_train)
        epoch_losses = []
        for start in tqdm(range(0, len(idx_train), args.batch_size), desc=f"epoch {epoch}", leave=False):
            batch_idx = idx_train[start : start + args.batch_size]
            if len(batch_idx) == 0:
                continue
            xy_batch = torch.tensor(coords_01[batch_idx], dtype=torch.float32, device=device)
            expr_batch = torch.tensor(gene_expr[batch_idx], dtype=torch.float32, device=device)
            with autocast(enabled=True):
                pred = F.relu(model.forward_block(xy_batch, gene_idx))
                loss = weighted_huber_loss(pred, expr_batch)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))
        with torch.no_grad(), autocast(enabled=True):
            pred_val = F.relu(model.forward_block(coords_val_gpu, gene_idx))
            val_loss = weighted_huber_loss(pred_val, torch.tensor(expr_val, dtype=torch.float32, device=device)).item()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    timer.end_step("training")

    timer.start_step("reconstruct")
    recon = []
    with torch.no_grad(), autocast(enabled=True):
        for start in tqdm(range(0, n_cells, 4096), desc="recon"):
            batch_coords = torch.tensor(coords_01[start : start + 4096], dtype=torch.float32, device=device)
            recon.append(F.relu(model.full_reconstruct(batch_coords)).cpu().numpy())
    reconstructed_expr = np.vstack(recon)
    timer.end_step("reconstruct")

    np.save(args.out_dir / "inr_reconstructed_expression.npy", reconstructed_expr)
    np.save(args.out_dir / "coords_original.npy", coords)
    np.save(args.out_dir / "coords_01.npy", coords_01)
    torch.save(model.state_dict(), args.out_dir / "inr_tucker_model.pth")
    save_json(
        args.out_dir / "inr_training_info.json",
        {
            "n_cells": int(n_cells),
            "n_genes": int(n_genes),
            "best_val_loss": float(best_val),
            "epochs": len(train_losses),
        },
    )

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "training_curves.png", dpi=150)
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
