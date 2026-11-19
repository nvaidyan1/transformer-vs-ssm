"""
Regression variants for the filtering tasks (plan2.md 5.2): continuous inputs through a
linear projection instead of an embedding, a linear head with one output, MSE loss.

  RegMamba  the custom Mamba blocks (src/models/mamba.py), pre-LN residual. Selective by
            default; forward(..., input_independent=True) is the non-selective ablation
            (Delta/B/C replaced by learned constants, exactly as in MambaSSM).
  RegGRU    a gated recurrent baseline (torch.nn.GRU), the "non-strawman" comparison.
"""

import torch
import torch.nn as nn

from src.models.mamba import MambaBlock


class RegMamba(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 32, n_layers: int = 1, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, scan: str = "sequential"):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand, 0.0, scan=scan)
                                     for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, u: torch.Tensor, input_independent: bool = False,
                return_delta: bool = False):
        """u: (B, T, in_dim) -> estimates (B, T) [, list of Delta (B, T, d_inner)]"""
        h = self.in_proj(u)
        deltas = []
        for blk in self.blocks:
            h, delta = blk(h, return_delta=return_delta, input_independent=input_independent)
            deltas.append(delta)
        out = self.head(self.ln_f(h)).squeeze(-1)
        return (out, deltas) if return_delta else out


class RegGRU(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 32, n_layers: int = 1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(d_model, 1)

    def forward(self, u: torch.Tensor, **_ignored):
        h, _ = self.gru(self.in_proj(u))
        return self.head(h).squeeze(-1)


def build_regression(arch: str, in_dim: int, d_model: int = 32, n_layers: int = 1,
                     scan: str = "parallel"):
    if arch in ("mamba", "mamba_ablated"):
        return RegMamba(in_dim, d_model=d_model, n_layers=n_layers, scan=scan)
    if arch == "gru":
        return RegGRU(in_dim, d_model=d_model, n_layers=n_layers)
    raise ValueError(arch)
