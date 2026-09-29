"""Forward + gradient parity against the official Mamba reference scan.

Recommended (external review, 2026-09-29) as a gate that should run BEFORE
interpreting any training result: if the recurrence/backprop implementation itself
disagrees with the reference, no downstream training result is interpretable, no
matter how it's initialised or optimized. This test answers that question directly,
independent of init, LR, or task.

Oracle: tests/vendor/selective_scan_ref.py, vendored verbatim (Apache-2.0) from
https://github.com/state-spaces/mamba -- Copyright (c) 2023, Tri Dao, Albert Gu.

Layout note: the reference uses (batch, dim, length); src.models.mamba.selective_scan
uses (batch, length, dim). B_ssm/C_ssm here are shared across the dim axis (shape
(batch, N, length) once transposed), matching the reference's `B.dim() == 3` branch --
this is the same layout MambaSSM actually uses (see src/models/mamba.py, x_proj
output), not a simplification made for this test.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.models.mamba import selective_scan
from vendor.selective_scan_ref import selective_scan_ref

TOL_FWD = 1e-4
TOL_GRAD = 1e-3   # gradients accumulate more float32 rounding than the forward pass


def _random_inputs(batch, E, N, T, seed, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    delta_pre = torch.randn(batch, T, E, generator=g)
    delta = torch.nn.functional.softplus(delta_pre)
    A = -torch.exp(torch.randn(E, N, generator=g) * 0.5)          # negative real, as in the model
    B_ssm = torch.randn(batch, T, N, generator=g) * 0.5
    C_ssm = torch.randn(batch, T, N, generator=g) * 0.5
    u = torch.randn(batch, T, E, generator=g)
    D = torch.randn(E, generator=g)
    z = torch.randn(batch, T, E, generator=g)
    tensors = [delta, A, B_ssm, C_ssm, u, D, z]
    if requires_grad:
        tensors = [t.clone().requires_grad_(True) for t in tensors]
    return tensors


def _run_ours(delta, A, B_ssm, C_ssm, u, D, z, scan):
    y = selective_scan(delta, A, B_ssm, C_ssm, u, scan=scan)
    out = y + D.unsqueeze(0).unsqueeze(0) * u
    out = out * torch.nn.functional.silu(z)
    return out  # (B, T, E)


def _run_reference(delta, A, B_ssm, C_ssm, u, D, z):
    # Reference layout is (batch, dim, length): transpose T<->E throughout.
    out = selective_scan_ref(
        u=u.transpose(1, 2), delta=delta.transpose(1, 2), A=A,
        B=B_ssm.transpose(1, 2), C=C_ssm.transpose(1, 2),
        D=D, z=z.transpose(1, 2), delta_softplus=False,
    )
    return out.transpose(1, 2)  # back to (B, T, E)


def _check_one(scan, seed, batch=2, E=8, N=4, T=16):
    delta, A, B_ssm, C_ssm, u, D, z = _random_inputs(batch, E, N, T, seed)
    y_ours = _run_ours(delta, A, B_ssm, C_ssm, u, D, z, scan=scan)
    y_ref = _run_reference(delta, A, B_ssm, C_ssm, u, D, z)

    fwd_diff = (y_ours - y_ref).abs().max().item()
    assert fwd_diff < TOL_FWD, f"[{scan}] forward diff {fwd_diff:.2e} exceeds {TOL_FWD}"

    loss_ours = y_ours.sum()
    loss_ref = y_ref.sum()
    grads_ours = torch.autograd.grad(loss_ours, [delta, A, B_ssm, C_ssm, u, D, z])
    grads_ref = torch.autograd.grad(loss_ref, [delta, A, B_ssm, C_ssm, u, D, z])

    names = ["delta", "A", "B_ssm", "C_ssm", "u", "D", "z"]
    for name, go, gr in zip(names, grads_ours, grads_ref):
        diff = (go - gr).abs().max().item()
        assert diff < TOL_GRAD, f"[{scan}] grad[{name}] diff {diff:.2e} exceeds {TOL_GRAD}"
    return fwd_diff


def test_sequential_scan_matches_reference_forward_and_gradients():
    for seed in range(5):
        _check_one("sequential", seed)


def test_parallel_scan_matches_reference_forward_and_gradients():
    for seed in range(5):
        _check_one("parallel", seed)


def test_sequential_and_parallel_agree_with_each_other():
    """Belt and suspenders: if both already independently match the reference,
    they must also match each other -- catches a shared, non-reference-caught bug."""
    delta, A, B_ssm, C_ssm, u, D, z = _random_inputs(2, 8, 4, 16, seed=0)
    y_seq = _run_ours(delta, A, B_ssm, C_ssm, u, D, z, scan="sequential")
    y_par = _run_ours(delta, A, B_ssm, C_ssm, u, D, z, scan="parallel")
    assert torch.allclose(y_seq, y_par, atol=1e-4)


def test_larger_shapes_still_agree():
    """Closer to the real model's dimensions (d_inner~248, d_state=16)."""
    diff = _check_one("sequential", seed=1, batch=4, E=64, N=16, T=32)
    assert diff < TOL_FWD
