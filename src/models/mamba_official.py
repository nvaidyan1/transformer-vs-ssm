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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mamba import (DEFAULT_EMB_INIT_STD, DEFAULT_OUT_PROJ_DEPTH_SCALE,
                              DT_MAX, DT_MIN, apply_model_init)


def _official_mixer_cls():
    from mamba_ssm.modules.mamba_simple import Mamba as OfficialMamba
    return OfficialMamba


def _official_scan_fn():
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    return selective_scan_fn


def _official_causal_conv1d_fn():
    """The causal_conv1d kernel the official slow path uses, or None if not installed
    (the official slow path then falls back to nn.Conv1d, as we do)."""
    try:
        from mamba_ssm.modules import mamba_simple
        return mamba_simple.causal_conv1d_fn
    except Exception:
        return None


def add_ablation_params(mixer: nn.Module):
    """Register the ablation-only constants on an official mixer, mirroring
    MambaSSM.fixed_delta_pre / fixed_B / fixed_C (same shapes, same init)."""
    d_inner, d_state = mixer.A_log.shape
    dt = torch.exp(torch.rand(d_inner) * (math.log(DT_MAX) - math.log(DT_MIN))
                   + math.log(DT_MIN))
    mixer.register_parameter("fixed_delta_pre",
                             nn.Parameter(dt + torch.log(-torch.expm1(-dt))))
    mixer.register_parameter("fixed_B", nn.Parameter(torch.randn(d_state) * 0.1))
    mixer.register_parameter("fixed_C", nn.Parameter(torch.randn(d_state) * 0.1))


def ablated_mixer_forward(mixer: nn.Module, hidden_states: torch.Tensor,
                          scan_fn=None, conv_fn=None) -> torch.Tensor:
    """The official mixer's slow path (mamba_simple.Mamba.forward with
    use_fast_path=False, no inference cache), line for line, except that Delta, B and C
    are the learned constants fixed_delta_pre / fixed_B / fixed_C broadcast over batch
    and time instead of projections of the input: x_proj and dt_proj are never called.
    Still calls the official selective_scan_fn (or `scan_fn`, tests only), which applies
    the D skip and the silu(z) gate exactly as in the selective path.

    Mirrors MambaSSM.forward(input_independent=True): delta = softplus(fixed_delta_pre)
    per channel, B = fixed_B and C = fixed_C shared across channels.
    """
    scan_fn = scan_fn or _official_scan_fn()
    batch, seqlen, _ = hidden_states.shape
    xz = (hidden_states @ mixer.in_proj.weight.t()).transpose(1, 2)     # (b, 2E, l)
    if mixer.in_proj.bias is not None:
        xz = xz + mixer.in_proj.bias.to(dtype=xz.dtype)[:, None]
    A = -torch.exp(mixer.A_log.float())
    x, z = xz.chunk(2, dim=1)
    if conv_fn is None:
        x = F.silu(mixer.conv1d(x)[..., :seqlen])
    else:
        x = conv_fn(x=x, weight=mixer.conv1d.weight.squeeze(1), bias=mixer.conv1d.bias,
                    activation="silu")
    d_inner, d_state = A.shape
    dt = mixer.fixed_delta_pre[None, :, None].expand(batch, d_inner, seqlen).contiguous()
    B = mixer.fixed_B[None, :, None].expand(batch, d_state, seqlen).contiguous()
    C = mixer.fixed_C[None, :, None].expand(batch, d_state, seqlen).contiguous()
    y = scan_fn(x, dt, A, B, C, mixer.D.float(), z=z, delta_bias=None,
                delta_softplus=True)
    return mixer.out_proj(y.transpose(1, 2))


class OracleBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, layer_idx, dropout=0.0,
                 use_fast_path=True, mixer_cls=None, scan_fn=None):
        super().__init__()
        mixer_cls = mixer_cls or _official_mixer_cls()
        self.scan_fn = scan_fn
        self.norm = nn.LayerNorm(d_model)
        self.mixer = mixer_cls(d_model=d_model, d_state=d_state, d_conv=d_conv,
                               expand=expand, layer_idx=layer_idx,
                               use_fast_path=use_fast_path)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, input_independent: bool = False):
        h = self.norm(x)
        if input_independent:
            conv_fn = None if self.scan_fn is not None else _official_causal_conv1d_fn()
            h = ablated_mixer_forward(self.mixer, h, scan_fn=self.scan_fn, conv_fn=conv_fn)
        else:
            h = self.mixer(h)
        return x + self.drop(h)


class OracleMamba(nn.Module):
    """Same constructor arguments as src.models.mamba.Mamba, minus `scan`, plus
    `use_fast_path` (fused mamba_inner_fn vs separate kernels; bit-identical in
    the oracle notebook) and `mixer_cls` / `scan_fn` (tests only: stand-ins for the
    official mixer and selective_scan_fn on machines where mamba-ssm cannot be imported).

    forward(x, input_independent=True) is the non-selective ablation (plan2.md 1.2a):
    every block runs ablated_mixer_forward, the official slow path with Delta/B/C
    replaced by learned constants. The constants are registered on each mixer as
    fixed_delta_pre / fixed_B / fixed_C (1,120 ablation-only params at mamba_small,
    the same as the custom model's), created after every other parameter so that the
    init of the rest of the model at a given seed is unchanged."""

    def __init__(self, vocab_size, n_layers, d_model, d_state, d_conv, expand,
                 dropout=0.0, use_fast_path=True,
                 emb_init_std=DEFAULT_EMB_INIT_STD,
                 out_proj_depth_scale=DEFAULT_OUT_PROJ_DEPTH_SCALE,
                 mixer_cls=None, scan_fn=None):
        super().__init__()
        mixer_cls = mixer_cls or _official_mixer_cls()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            OracleBlock(d_model, d_state, d_conv, expand, layer_idx=i,
                        dropout=dropout, use_fast_path=use_fast_path,
                        mixer_cls=mixer_cls, scan_fn=scan_fn)
            for i in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying, matching our model

        self.n_layers = n_layers
        self.emb_init_std = emb_init_std
        self.out_proj_depth_scale = out_proj_depth_scale
        apply_model_init(self, n_layers, emb_init_std, out_proj_depth_scale)
        # last, so the RNG stream for everything above is as it was before 1.2a
        for block in self.blocks:
            add_ablation_params(block.mixer)

    def forward(self, x: torch.Tensor, input_independent: bool = False):
        h = self.tok_emb(x)
        for block in self.blocks:
            h = block(h, input_independent=input_independent)
        return self.head(self.ln_f(h))


def copy_custom_to_official(custom: nn.Module, official: nn.Module) -> list:
    """Inverse of copy_official_to_custom, including the ablation-only params (both
    models now have them). Returns the official parameter names written."""
    official_params = dict(official.named_parameters())
    written = []
    with torch.no_grad():
        for name, p in custom.named_parameters():
            target = name.replace(".ssm.", ".mixer.")
            if target not in official_params:
                raise KeyError(f"no official parameter for custom {name!r}")
            if official_params[target].shape != p.shape:
                raise ValueError(f"shape mismatch for {name!r}")
            official_params[target].copy_(p)
            written.append(target)
    missing = [n for n in official_params if n not in written]
    if missing:
        raise KeyError(f"official parameters not covered by the custom model: {missing}")
    return written


def copy_official_to_custom(official: nn.Module, custom: nn.Module) -> list:
    """Copy every parameter of an OracleMamba into a custom Mamba of the same config.

    Names map by `.mixer.` -> `.ssm.`; the custom model's ablation-only fixed_*
    params have no official counterpart and are left untouched (they are unused
    unless --ablate). Raises if any official parameter has no match or a different
    shape. Returns the list of custom parameter names that were written.
    """
    custom_params = dict(custom.named_parameters())
    written = []
    with torch.no_grad():
        for name, p in official.named_parameters():
            target = name.replace(".mixer.", ".ssm.")
            if target not in custom_params:
                raise KeyError(f"no custom parameter for official {name!r}")
            if custom_params[target].shape != p.shape:
                raise ValueError(f"shape mismatch for {name!r}: "
                                 f"{tuple(p.shape)} vs {tuple(custom_params[target].shape)}")
            custom_params[target].copy_(p)
            written.append(target)
    missing = [n for n in custom_params if n not in written and ".fixed_" not in n]
    if missing:
        raise KeyError(f"custom parameters not covered by the official model: {missing}")
    return written
