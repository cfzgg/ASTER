import numpy as np
import torch
import torch.nn as nn
from torch.nn import Embedding, PReLU


class SineLayer(nn.Module):
    """SIREN-style sinusoidal activation layer."""

    def __init__(self, in_features: int, out_features: int, omega_0: float = 1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        bound = np.sqrt(6.0 / (in_features + out_features))
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class ASTER_NTD(nn.Module):
    def __init__(
        self,
        n_g: int,
        rank_x: int = 128,
        rank_y: int = 128,
        rank_g: int = 128,
        mid_feature: int = 200,
    ):
        super().__init__()
        self.X_net = nn.Sequential(
            SineLayer(1, mid_feature),
            SineLayer(mid_feature, mid_feature),
            nn.Linear(mid_feature, rank_x),
            PReLU(init=0.9),
        )
        self.Y_net = nn.Sequential(
            SineLayer(1, mid_feature),
            SineLayer(mid_feature, mid_feature),
            nn.Linear(mid_feature, rank_y),
            PReLU(init=0.9),
        )
        self.G_net = nn.Sequential(
            Embedding(n_g, rank_g),
            nn.Linear(rank_g, rank_g),
            PReLU(init=0.9),
        )

    def forward(
        self,
        core: torch.Tensor,
        g_idx: torch.Tensor,
        X_input: torch.Tensor,
        Y_input: torch.Tensor,
    ) -> torch.Tensor:
        x = self.X_net(X_input)
        y = self.Y_net(Y_input)
        g = self.G_net(g_idx)
        out = torch.einsum("abc,ic,ja,kb->ijk", core, g, x, y)
        return out.relu_()

