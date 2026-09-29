"""The two scan implementations must be interchangeable.

`scan="parallel"` exists so the implementation choice can be revisited per shape. It is
only safe to switch if it is numerically equivalent to the sequential loop, so that is
asserted here rather than assumed.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model
from src.models.mamba import MambaSSM

SMALL = dict(n_layers=2, d_model=32, d_state=8, d_conv=4, expand=2, dropout=0.0)


def _pair():
    torch.manual_seed(0)
    a = build_model("mamba", vocab_size=64, scan="sequential", **SMALL).eval()
    torch.manual_seed(0)
    b = build_model("mamba", vocab_size=64, scan="parallel", **SMALL).eval()
    return a, b


def test_scans_agree_on_forward():
    seq, par = _pair()
    x = torch.randint(0, 64, (4, 32))
    with torch.no_grad():
        assert torch.allclose(seq(x), par(x), atol=1e-4)


def test_scans_agree_under_ablation():
    seq, par = _pair()
    x = torch.randint(0, 64, (4, 32))
    with torch.no_grad():
        a = seq(x, input_independent=True)
        b = par(x, input_independent=True)
    assert torch.allclose(a, b, atol=1e-4)


def test_scans_agree_on_gradients():
    seq, par = _pair()
    x = torch.randint(0, 64, (4, 32))
    grads = []
    for m in (seq, par):
        m.train()
        m.zero_grad()
        m(x).sum().backward()
        grads.append(torch.cat([p.grad.flatten() for _, p in sorted(m.named_parameters())
                                if p.grad is not None]))
    assert torch.allclose(grads[0], grads[1], atol=1e-3), "scans disagree on gradients"


def test_default_is_sequential():
    """Changing the default would silently alter the enwik8 configuration."""
    m = build_model("mamba", vocab_size=64, **SMALL)
    assert all(b.ssm.scan == "sequential" for b in m.blocks)


def test_invalid_scan_rejected():
    with pytest.raises(ValueError):
        MambaSSM(d_model=32, d_state=8, d_conv=4, expand=2, scan="fft")
