from __future__ import annotations

import torch
import torch.nn as nn


class LocalKNNAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, neighbors, all_x):
        b, dim = x.shape
        k = neighbors.shape[1]
        neighbor_feat = all_x[neighbors]
        x_with_neighbors = torch.cat([x.unsqueeze(1), neighbor_feat], dim=1)
        qkv = self.qkv(x_with_neighbors).reshape(b, k + 1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k_, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k_.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = attn @ v
        return self.proj(out.transpose(1, 2).reshape(b, k + 1, dim)[:, 0, :])


class BCAM(nn.Module):
    def __init__(self, gene_dim: int, hist_dim: int, hidden_dim: int = 512, latent_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.gene_proj = nn.Sequential(nn.Linear(gene_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.hist_proj = nn.Sequential(nn.Linear(hist_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.ca1_g2h = LocalKNNAttention(hidden_dim, dropout=dropout)
        self.ca1_h2g = LocalKNNAttention(hidden_dim, dropout=dropout)
        self.ca2_g2h = LocalKNNAttention(hidden_dim, dropout=dropout)
        self.ca2_h2g = LocalKNNAttention(hidden_dim, dropout=dropout)
        self.ca3_g2h = LocalKNNAttention(hidden_dim, dropout=dropout)
        self.ca3_h2g = LocalKNNAttention(hidden_dim, dropout=dropout)
        self.ln1_g = nn.LayerNorm(hidden_dim)
        self.ln1_h = nn.LayerNorm(hidden_dim)
        self.ln2_g = nn.LayerNorm(hidden_dim)
        self.ln2_h = nn.LayerNorm(hidden_dim)
        self.ln3_g = nn.LayerNorm(hidden_dim)
        self.ln3_h = nn.LayerNorm(hidden_dim)
        self.ffn_g = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.ffn_h = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim))
        self.recon_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, gene_dim), nn.Softplus())

    def forward(self, gene, hist, neighbors, all_gene, all_hist):
        g = self.gene_proj(gene)
        h = self.hist_proj(hist)
        all_g = self.gene_proj(all_gene)
        all_h = self.hist_proj(all_hist)
        g1 = self.ln1_g(g + self.ca1_h2g(g, neighbors, all_h))
        h1 = self.ln1_h(h + self.ca1_g2h(h, neighbors, all_g))
        g2 = self.ln2_g(g1 + self.ca2_h2g(g1, neighbors, all_h))
        h2 = self.ln2_h(h1 + self.ca2_g2h(h1, neighbors, all_g))
        g3 = self.ln3_g(g2 + self.ca3_h2g(g2, neighbors, all_h))
        h3 = self.ln3_h(h2 + self.ca3_g2h(h2, neighbors, all_g))
        latent = self.fusion(torch.cat([g3 + self.ffn_g(g3), h3 + self.ffn_h(h3)], dim=1))
        return latent, self.recon_head(latent)
