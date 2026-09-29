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


def test_ablation_is_genuinely_input_independent():
    """Regression test for the exact flaw an external reviewer found (2026-09-29):
    the original ablation averaged delta/B/C over the time axis, so the 'fixed'
    value used at an early position still depended on later positions in the same
    sequence -- not actually input-independent. This asserts the current
    (learned-constant) ablation has zero gradient dependency on the input at all,
    at every position, not just zero *position-wise variation*."""
    m = _model().eval()
    x = torch.randn(2, 16, 32, requires_grad=True)
    h = m.tok_emb.weight  # unused; keep x as the leaf we test
    ssm = m.blocks[0].ssm
    x_proj = torch.randn(2, 16, 32, requires_grad=True)

    _, delta = ssm(x_proj, return_delta=True, input_independent=True)
    # delta must not require grad w.r.t. x_proj at all under ablation -- there is
    # no path from the input to delta/B/C in this mode.
    grad = torch.autograd.grad(delta.sum(), x_proj, retain_graph=True,
                               allow_unused=True)[0]
    assert grad is None or torch.equal(grad, torch.zeros_like(grad)), (
        "ablated delta still depends on the input -- the leak has come back")


def test_ablation_uses_learned_constants_not_dt_proj():
    """dt_proj/x_proj must not even be called under ablation (not just unused in
    the output) -- verified via a forward hook that would fire if they were."""
    m = _model().eval()
    ssm = m.blocks[0].ssm
    called = []
    h1 = ssm.dt_proj.register_forward_hook(lambda *a: called.append("dt_proj"))
    h2 = ssm.x_proj.register_forward_hook(lambda *a: called.append("x_proj"))
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        ssm(x, input_independent=True)
    h1.remove(); h2.remove()
    assert called == [], f"dt_proj/x_proj were called under ablation: {called}"


def test_ablation_params_receive_gradients():
    """fixed_delta_pre/fixed_B/fixed_C must actually be trainable -- otherwise the
    ablation is stuck at a fixed random init, which would be an unfairly weak
    control rather than a genuine time-invariant-SSM baseline."""
    m = _model().train()
    ssm = m.blocks[0].ssm
    x = torch.randint(0, 256, (2, 16))
    m.zero_grad()
    out = m(x, input_independent=True)
    out.sum().backward()
    for name, p in [("fixed_delta_pre", ssm.fixed_delta_pre),
                    ("fixed_B", ssm.fixed_B), ("fixed_C", ssm.fixed_C)]:
        assert p.grad is not None, f"{name} got no gradient under ablation"
        assert p.grad.abs().sum() > 0, f"{name} gradient is all-zero"
