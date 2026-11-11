"""
Mamba built on the official mamba-ssm mixer (compiled CUDA/Triton kernels).

This is the model reported numbers come from (see recap.md section 7). The
from-scratch model in src/models/mamba.py is the explanatory reference. The two are
meant to differ ONLY in the mixer implementation, so everything around the mixer is
shared or identical:

  - block structure: pre-LN residual, x + dropout(mixer(LayerNorm(x)))
  - tied embedding / head, final LayerNorm, forward(x) -> logits
  - model-level init: apply_model_init() from src/models/mamba.py, same defaults
  - optimizer grouping: src/optim.py, matched by parameter-name suffix; the only
    naming difference is `blocks.i.mixer.*` here vs `blocks.i.ssm.*` there

Moved from notebooks/kaggle_mamba_ssm_oracle.ipynb (cell 4). That notebook's wrapper
applied no model-level init: embedding N(0, 1), out_proj unscaled. To reproduce it,
pass emb_init_std=1.0, out_proj_depth_scale=False. The defaults here follow the
reference language-model recipe instead (0.02 and depth-scaled), same as the custom
model.

mamba-ssm is imported lazily, on first construction, so this module (and the rest
of the repo, including the test suite) imports fine on machines without CUDA.
"""

import torch
import torch.nn as nn

from src.models.mamba import (DEFAULT_EMB_INIT_STD, DEFAULT_OUT_PROJ_DEPTH_SCALE,
                              apply_model_init)


def _official_mixer_cls():
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba
    return OfficialMamba


class OracleBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, layer_idx, dropout=0.0,
                 use_fast_path=True, mixer_cls=None):
        super().__init__()
        mixer_cls = mixer_cls or _official_mixer_cls()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = mixer_cls(d_model=d_model, d_state=d_state, d_conv=d_conv,
                               expand=expand, layer_idx=layer_idx,
                               use_fast_path=use_fast_path)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.mixer(self.norm(x)))


class OracleMamba(nn.Module):
    """Same constructor arguments as src.models.mamba.Mamba, minus `scan`, plus
    `use_fast_path` (fused mamba_inner_fn vs separate kernels; bit-identical in
    the oracle notebook) and `mixer_cls` (tests only: a stand-in for the official
    mixer on machines where mamba-ssm cannot be imported)."""

    def __init__(self, vocab_size, n_layers, d_model, d_state, d_conv, expand,
                 dropout=0.0, use_fast_path=True,
                 emb_init_std=DEFAULT_EMB_INIT_STD,
                 out_proj_depth_scale=DEFAULT_OUT_PROJ_DEPTH_SCALE,
                 mixer_cls=None):
        super().__init__()
        mixer_cls = mixer_cls or _official_mixer_cls()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            OracleBlock(d_model, d_state, d_conv, expand, layer_idx=i,
                        dropout=dropout, use_fast_path=use_fast_path,
                        mixer_cls=mixer_cls)
            for i in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying, matching our model

        self.n_layers = n_layers
        self.emb_init_std = emb_init_std
        self.out_proj_depth_scale = out_proj_depth_scale
        apply_model_init(self, n_layers, emb_init_std, out_proj_depth_scale)

    def forward(self, x: torch.Tensor):
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.ln_f(h))
