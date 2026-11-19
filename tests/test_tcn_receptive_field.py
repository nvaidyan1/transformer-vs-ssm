"""plan2.md 3.1: the TCN's receptive field, measured by gradients, equals the formula.

The gradient of the output at position t with respect to the input embedding at t-k is
exactly zero for k >= RF and non-zero at k = RF - 1. Checked for tcn_short (RF 127) and
tcn_small (RF 769) with their real configs."""

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models import build_model


def grad_reach(model, T):
    """|d out[T-1] / d emb[s]| summed over channels, for every source position s."""
    model.eval()
    emb = torch.randn(1, T, model.tok_emb.weight.shape[1], requires_grad=True)
    h = emb.transpose(1, 2)
    for block in model.blocks:
        h = block(h)
    out = model.head(model.ln_f(h.transpose(1, 2)))
    out[0, T - 1].sum().backward()
    return emb.grad[0].abs().sum(-1)            # (T,)


@pytest.mark.parametrize("name,rf", [("tcn_short", 127), ("tcn_small", 769)])
def test_gradient_receptive_field_matches_formula(name, rf):
    cfg = yaml.safe_load(open(ROOT / f"configs/recall/{name}.yaml"))
    torch.manual_seed(0)
    m = build_model("tcn", vocab_size=cfg["vocab_size"], **cfg["model"])
    assert m.receptive_field == rf
    T = rf + 20
    g = grad_reach(m, T)
    t = T - 1
    for k in range(rf, T):                      # at or beyond the RF: exactly zero
        assert g[t - k].item() == 0.0, (name, k)
    assert g[t - (rf - 1)].item() > 0.0         # just inside: non-zero
    print(f"{name}: receptive field {rf} (gradient-verified)")
