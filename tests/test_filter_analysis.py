"""plan2.md 5.2/5.3 code: regression models, effective gain, Delta by regime."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filter_analysis import delta_by_regime, effective_gain, pearson
from src.models.regression import build_regression
from src.tasks.filtering import constant_gain, generate


class ConstGainFilter(nn.Module):
    """x_hat_t = (1-k) x_hat_{t-1} + k y_t, as a differentiable module."""
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, u, **_):
        est, m = [], torch.zeros(u.shape[0], device=u.device)
        for t in range(u.shape[1]):
            m = m + self.k * (u[:, t, 0] - m)
            est.append(m)
        return torch.stack(est, 1)


def test_effective_gain_recovers_a_known_gain():
    u = torch.randn(3, 20, 2)
    g = effective_gain(ConstGainFilter(0.3), u)
    assert np.allclose(g, 0.3, atol=1e-6)


def test_constant_gain_module_matches_numpy_filter():
    u, *_ = generate("switching", 4, 30, seed=0)
    ref = constant_gain(u[..., 0].astype(float), 0.2)
    got = ConstGainFilter(0.2)(torch.from_numpy(u)).double().numpy()
    assert np.allclose(ref, got, atol=1e-5)


def test_pearson():
    a = np.arange(10.0)
    assert abs(pearson(a, 2 * a + 1) - 1) < 1e-12 and abs(pearson(a, -a) + 1) < 1e-12


def test_models_shapes_causality_and_ablation_switch():
    torch.manual_seed(0)
    u = torch.randn(2, 16, 2)
    for arch in ("mamba", "mamba_ablated", "gru"):
        for scan in ("sequential", "parallel"):
            m = build_regression(arch, 2, d_model=8, n_layers=2, scan=scan).eval()
            kw = {"input_independent": True} if arch == "mamba_ablated" else {}
            out = m(u, **kw)
            assert out.shape == (2, 16)
            u2 = u.clone(); u2[:, 10:] += 5.0                  # future change
            assert torch.allclose(m(u2, **kw)[:, :10], out[:, :10], atol=1e-5)
    m = build_regression("mamba", 2, d_model=8, n_layers=1)
    assert not torch.allclose(m(u), m(u, input_independent=True))


def test_delta_by_regime_shapes():
    torch.manual_seed(0)
    m = build_regression("mamba", 2, d_model=8, n_layers=2)
    u, x, r, flag = generate("switching", 4, 64, seed=1)
    out = delta_by_regime(m, torch.from_numpy(u), flag)
    assert len(out) == 2 and len(out[0]["low_per_channel"]) == 16
