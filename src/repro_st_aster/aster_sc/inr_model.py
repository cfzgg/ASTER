"""SIREN + low-rank Tucker-2 modules shared by the ASTER-SC INR stage.

These classes were previously defined inline inside ``reconstruct.py``. They are
factored out here so the Visium HD variant (``reconstruct_hd.py``) can reuse the
exact same architecture with its own training loop.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


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
    """Continuous field ``f(s, g) = xy_net(s) @ K @ g_emb^T``.

    ``k_init_gain`` selects the coupling-matrix initialisation. The single-cell
    workflows (breast cancer, gastric cancer) use ``gain=0.5``; the Visium HD
    colorectal workflow used the plain ``xavier_uniform_`` default, which is
    ``k_init_gain=None`` here.
    """

    def __init__(
        self,
        n_genes,
        r_s=512,
        r_g=256,
        hidden=512,
        depth=6,
        omega_0=1.0,
        p_dropout=0.1,
        k_init_gain: float | None = 0.5,
    ):
        super().__init__()
        self.xy_net = XYNetSIREN(hidden=hidden, depth=depth, out_dim=r_s, omega_0=omega_0, p_dropout=p_dropout)
        self.g_emb = nn.Embedding(n_genes, r_g)
        self.K = nn.Parameter(torch.empty(r_s, r_g))
        if k_init_gain is None:
            nn.init.xavier_uniform_(self.K)
        else:
            nn.init.xavier_uniform_(self.K, gain=k_init_gain)
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


def omega_ramp(epoch: int, max_epoch: int, omega_start: float, omega_end: float, ramp=(0.2, 0.6)) -> float:
    """Linear omega warm-up between two fractions of the training schedule."""
    a, b = ramp
    p = epoch / max_epoch
    if p <= a:
        return omega_start
    if p >= b:
        return omega_end
    return omega_start + (p - a) / (b - a) * (omega_end - omega_start)
