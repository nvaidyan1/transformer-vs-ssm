"""Delta (the SSM's selection timestep) must be initialised small.

The gate on Selective Copying (see PLAN.md) found Mamba stuck at chance accuracy.
Root cause: with a bias-free projection, softplus(pre-activation) lands at
softplus(0) = ln(2) = 0.693 by default. Combined with A in [-1, -16], that gives a
state half-life of about ONE step -- the model cannot retain information across a
64-token sequence regardless of training.

Fix: a dedicated dt_proj whose bias is set so softplus(bias) alone is log-uniform in
[DT_MIN, DT_MAX] (Gu & Dao's reference range), before any input-dependent term is
added. These tests exist because the fix was silently erased once already: after
MambaSSM.__init__ sets the bias correctly, Mamba._init_weights() re-initialises
every nn.Linear in the model afterward and would zero it again without an explicit
opt-out. If these tests regress, check that opt-out first.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model
from src.models.mamba import DT_MAX, DT_MIN, MambaSSM


def test_dt_proj_bias_is_no_reinit_marked():
    """The exact regression: Mamba._init_weights() must not touch dt_proj."""
    ssm = MambaSSM(d_model=64, d_state=16, d_conv=4, expand=2)
    assert getattr(ssm.dt_proj.weight, "_no_reinit", False) is True
    assert getattr(ssm.dt_proj.bias, "_no_reinit", False) is True


def test_dt_bias_gives_softplus_in_reference_range():
    """softplus(bias) alone, before any input contribution, must land in
    [DT_MIN, DT_MAX] for (nearly) every channel."""
    ssm = MambaSSM(d_model=128, d_state=16, d_conv=4, expand=2)
    sp_bias = torch.nn.functional.softplus(ssm.dt_proj.bias.detach())
    frac_in_range = ((sp_bias >= DT_MIN) & (sp_bias <= DT_MAX)).float().mean()
    assert frac_in_range > 0.99, f"only {frac_in_range:.2%} of dt biases in range"


def test_full_model_delta_survives_init_weights():
    """The regression specifically happened through the FULL model's __init__,
    not through MambaSSM in isolation -- test the actual construction path."""
    m = build_model("mamba", vocab_size=64, n_layers=4, d_model=124, d_state=16,
                    d_conv=4, expand=2, dropout=0.0).eval()
    x = torch.randint(0, 64, (8, 64))
    with torch.no_grad():
        _, deltas = m(x, return_delta=True)
    d = torch.cat([t.flatten() for t in deltas])
    assert d.median() < 10 * DT_MAX, (
        f"delta median {d.median():.4f} suggests dt_proj was reinitialised "
        f"(expect near [{DT_MIN}, {DT_MAX}], got median far above it)")


def test_state_can_survive_64_steps_at_init():
    """The concrete failure mode: at the old init, even the slowest state channel
    (A closest to 0) decayed to ~0 within 64 steps. Assert it no longer does."""
    m = build_model("mamba", vocab_size=64, n_layers=4, d_model=124, d_state=16,
                    d_conv=4, expand=2, dropout=0.0).eval()
    x = torch.randint(0, 64, (8, 64))
    with torch.no_grad():
        _, deltas = m(x, return_delta=True)
    dt_mean = torch.cat([t.flatten() for t in deltas]).mean().item()
    A = -torch.exp(m.blocks[0].ssm.A_log)
    slowest_A = A.max().item()  # closest to 0 -> slowest decay
    retained = math.exp(dt_mean * slowest_A) ** 64
    assert retained > 0.01, (
        f"slowest state channel retains only {retained:.2e} of its signal after "
        f"64 steps at init -- selective copying over that span is not learnable")
