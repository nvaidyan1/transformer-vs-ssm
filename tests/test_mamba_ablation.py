"""The selectivity ablation must be inert by default and functional when requested.

This is the control for the Selective Copying gate: a working selective implementation
must beat `input_independent=True` by a wide margin. If the flag silently changed the
default path, every downstream comparison would be invalid.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model

SMALL = dict(n_layers=2, d_model=32, d_state=8, d_conv=4, expand=2, dropout=0.0)


def _model():
    torch.manual_seed(0)
    return build_model("mamba", **SMALL)


def test_default_path_is_bit_identical():
    m = _model().eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        assert torch.equal(m(x), m(x, input_independent=False))


def test_ablation_changes_the_output_but_not_the_shape():
    m = _model().eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        base, abl = m(x), m(x, input_independent=True)
    assert base.shape == abl.shape
    assert not torch.allclose(base, abl), "ablation had no effect — selection may be dead"


def test_ablation_removes_position_wise_variation():
    """With selection frozen, delta must be constant along the sequence axis."""
    m = _model().eval()
    x = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        _, deltas = m(x, return_delta=True, input_independent=True)
    for d in deltas:
        spread = (d - d.mean(dim=1, keepdim=True)).abs().max().item()
        assert spread < 1e-5, f"delta still varies with position under ablation ({spread:.2e})"


def test_selective_delta_does_vary_with_position():
    m = _model().eval()
    x = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        _, deltas = m(x, return_delta=True)
    spread = max((d - d.mean(dim=1, keepdim=True)).abs().max().item() for d in deltas)
    assert spread > 1e-5, "delta is position-invariant even without the ablation"


def test_gradient_checkpoint_path_still_works():
    """Mamba.forward uses torch.utils.checkpoint in training mode; the extra
    argument must be threaded through it correctly."""
    m = _model().train()
    x = torch.randint(0, 256, (2, 16))
    for flag in (False, True):
        m.zero_grad()
        logits = m(x, input_independent=flag)
        logits.sum().backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads, f"no gradients flowed (input_independent={flag})"
        assert all(torch.isfinite(g).all() for g in grads)
