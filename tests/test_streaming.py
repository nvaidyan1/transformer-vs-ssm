"""plan2.md 2.2: token-by-token steppers reproduce the full-sequence forward, so the bytes
they hold are the real carried state; analytic byte counts match the measured ones."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model
from src.streaming import MambaStepper, TCNStepper, TransformerStepper, analytic_bytes

V, T = 32, 40


def models():
    torch.manual_seed(0)
    return {
        "transformer": (build_model("transformer", vocab_size=V, n_layers=2, d_model=16,
                                    n_heads=2, d_ff=32, dropout=0.0, max_seq_len=64),
                        TransformerStepper),
        "tcn": (build_model("tcn", vocab_size=V, n_layers=4, d_model=16, kernel_size=3,
                            dropout=0.0), TCNStepper),
        "mamba": (build_model("mamba", vocab_size=V, n_layers=2, d_model=16, d_state=4,
                              d_conv=4, expand=2, dropout=0.0, scan="sequential",
                              grad_checkpoint=False), MambaStepper),
    }


@pytest.mark.parametrize("arch", ["transformer", "tcn", "mamba"])
def test_stepping_matches_full_forward(arch):
    model, Stepper = models()[arch]
    model.eval()
    x = torch.randint(0, V, (2, T))
    with torch.no_grad():
        full = model(x)
    st = Stepper(model)
    steps = torch.stack([st.step(x[:, t]) for t in range(T)], dim=1)
    assert (steps - full).abs().max().item() < 1e-4


@pytest.mark.parametrize("arch", ["transformer", "tcn", "mamba"])
def test_measured_bytes_equal_analytic(arch):
    model, Stepper = models()[arch]
    st = Stepper(model)
    x = torch.randint(0, V, (1, T))
    for t in range(T):
        st.step(x[:, t])
        assert st.state_bytes() == analytic_bytes(arch, model, t + 1), (arch, t)


def test_state_growth_shapes():
    ms = models()
    tf, tcn, mb = (ms[a][0] for a in ("transformer", "tcn", "mamba"))
    # KV cache linear; TCN saturates at its padding; Mamba constant after d_conv - 1 tokens
    assert analytic_bytes("transformer", tf, 20) == 2 * analytic_bytes("transformer", tf, 10)
    rf_pad = sum(b.conv.padding for b in tcn.blocks)
    assert analytic_bytes("tcn", tcn, 10_000) == analytic_bytes("tcn", tcn, rf_pad + 5)
    assert analytic_bytes("mamba", mb, 10_000) == analytic_bytes("mamba", mb, 4)


def test_sdpa_matches_naive_attention():
    torch.manual_seed(0)
    kw = dict(vocab_size=V, n_layers=2, d_model=16, n_heads=2, d_ff=32, dropout=0.0,
              max_seq_len=64)
    a = build_model("transformer", **kw).eval()
    b = build_model("transformer", attn_impl="sdpa", **kw).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randint(0, V, (2, T))
    with torch.no_grad():
        assert (a(x) - b(x)).abs().max().item() < 1e-5
