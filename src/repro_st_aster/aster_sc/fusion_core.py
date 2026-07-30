"""FusionNet: the bidirectional cross-attention fusion head used by the
colorectal cancer Visium HD workflow (Fig. 4).

This is a *different* architecture from :class:`repro_st_aster.uni_bcam.bcam_core.BCAM`,
which the single-cell workflows (breast cancer, gastric cancer) use. Both fuse
INR-reconstructed expression with UNI-2 morphology over a spatial neighbourhood,
but they differ in attention layout and in how the clustering latent is taken.

.. warning::
   ``FusionNet.forward`` returns ``(pred, latent)`` whereas ``BCAM.forward``
   returns ``(latent, recon)`` -- the order is **reversed**. Check the call site.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """Multi-head cross-attention with independent q / k / v / out projections."""

    def __init__(self, embed_dim: int, q_dim: int, kv_dim: int, num_heads: int = 8):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.query_proj = nn.Linear(q_dim, embed_dim)
        self.key_proj = nn.Linear(kv_dim, embed_dim)
        self.value_proj = nn.Linear(kv_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        b, n_q, _ = query.shape
        n_k = key.shape[1]
        q = self.query_proj(query).view(b, n_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key_proj(key).view(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value_proj(value).view(b, n_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(b, n_q, self.embed_dim)
        return self.out_proj(context)


class CrossAttentionModel(nn.Module):
    """Three cross-attention layers: his<-gene, gene<-his, then fuse the two."""

    def __init__(self, embed_dim: int, his_dim: int, ge_dim: int, num_heads: int = 8):
        super().__init__()
        self.ca1 = CrossAttentionLayer(embed_dim, q_dim=his_dim, kv_dim=ge_dim, num_heads=num_heads)
        self.ca2 = CrossAttentionLayer(embed_dim, q_dim=ge_dim, kv_dim=his_dim, num_heads=num_heads)
        self.ca3 = CrossAttentionLayer(embed_dim, q_dim=embed_dim, kv_dim=embed_dim, num_heads=num_heads)

    def forward(self, a, b):
        """``a`` = histology features (B, N, C_his); ``b`` = gene features (B, N, C_gene)."""
        c = self.ca1(a, b, b)
        d = self.ca2(b, a, a)
        return self.ca3(c, d, d)


class FusionNet(nn.Module):
    """Two stacked cross-attention blocks + an output MLP over the centre token.

    The clustering latent is the 512-d activation *before* the final linear layer
    of ``output_mlp`` (iSTAR-style), so the gene reconstruction head stays intact.

    Neighbour convention: the centre spot must be neighbour 0 (``x[:, 0, :]``), i.e.
    the KNN graph must **include self**. See ``fusion.run_fusion``.
    """

    def __init__(self, uni_dim: int, inr_dim: int, out_dim: int, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.uni_dim = uni_dim
        self.inr_dim = inr_dim
        self.out_dim = out_dim
        self.mca1 = CrossAttentionModel(embed_dim=embed_dim, his_dim=uni_dim, ge_dim=inr_dim, num_heads=num_heads)
        self.l1 = nn.Linear(embed_dim, embed_dim)
        self.mca2 = CrossAttentionModel(embed_dim=embed_dim, his_dim=embed_dim, ge_dim=inr_dim, num_heads=num_heads)
        self.l2 = nn.Linear(embed_dim, embed_dim)
        self.output_mlp = nn.Sequential(
            nn.Linear(embed_dim + uni_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, uni2, inr):
        """``uni2`` (B, K, C_uni), ``inr`` (B, K, G) -> ``(pred (B, G), latent (B, 512))``."""
        x = self.mca1(uni2, inr)
        x = F.relu(self.l1(x) + x, inplace=True)
        x = self.mca2(x, inr)
        x = F.relu(self.l2(x) + x, inplace=True)

        x_center = x[:, 0, :]
        his_center = uni2[:, 0, :]
        fusion = torch.cat([x_center, his_center], dim=-1)

        latent = fusion
        for layer in self.output_mlp[:-1]:
            latent = layer(latent)
        pred = self.output_mlp[-1](latent)
        return pred, latent
