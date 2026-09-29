"""ssm_lr_scale must actually reach the optimizer step, every step.

The trap: writing the scheduled LR straight into every group's g['lr'] (a natural
thing to do for a warmup+cosine schedule) silently overwrites the per-group
multiplier set at construction. These tests exercise the exact pattern
src/train_task.py uses, not just build_param_groups() in isolation.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model
from src.optim import build_param_groups, is_ssm_selectivity


def test_selectivity_params_identified_correctly():
    m = build_model("mamba", vocab_size=32, n_layers=2, d_model=32, d_state=8,
                    d_conv=4, expand=2, dropout=0.0)
    hit = {n for n, _ in m.named_parameters() if is_ssm_selectivity(n)}
    assert any(n.endswith("dt_proj.weight") for n in hit)
    assert any(n.endswith("dt_proj.bias") for n in hit)
    assert any(n.endswith("A_log") for n in hit)
    assert any(n.endswith("x_proj.weight") for n in hit)
    # D and in/out/conv projections must NOT be boosted -- their gradients were
    # already normal-scale in the diagnostic that motivated this.
    assert not any(n.endswith(".D") for n in hit)
    assert not any("out_proj" in n or "in_proj" in n or "conv1d" in n for n in hit)


def test_every_param_in_exactly_one_group():
    for arch, kw in [("transformer", dict(n_layers=2, d_model=32, n_heads=2, d_ff=64,
                                          dropout=0.0, max_seq_len=32)),
                     ("tcn", dict(n_layers=3, d_model=32, kernel_size=3, dropout=0.0)),
                     ("mamba", dict(n_layers=2, d_model=32, d_state=8, d_conv=4,
                                    expand=2, dropout=0.0))]:
        m = build_model(arch, vocab_size=32, **kw)
        groups = build_param_groups(m, weight_decay=0.1, ssm_lr_scale=10.0)
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        covered = sum(p.numel() for g in groups for p in g["params"])
        assert total == covered, f"{arch}: {covered} != {total}"


def test_non_mamba_architectures_have_no_boosted_group():
    for arch, kw in [("transformer", dict(n_layers=2, d_model=32, n_heads=2, d_ff=64,
                                          dropout=0.0, max_seq_len=32)),
                     ("tcn", dict(n_layers=3, d_model=32, kernel_size=3, dropout=0.0))]:
        m = build_model(arch, vocab_size=32, **kw)
        groups = build_param_groups(m, weight_decay=0.1, ssm_lr_scale=10.0)
        assert all(g["lr_scale"] == 1.0 for g in groups), (
            f"{arch} has a boosted group but no SSM selectivity parameters exist")


def test_lr_scale_survives_the_scheduler_step_pattern():
    """Reproduce src/train_task.py's loop exactly: base_lr changes every step, and
    every group's lr must equal base_lr * that group's own lr_scale, every time."""
    m = build_model("mamba", vocab_size=32, n_layers=2, d_model=32, d_state=8,
                    d_conv=4, expand=2, dropout=0.0)
    groups = build_param_groups(m, weight_decay=0.1, ssm_lr_scale=25.0)
    opt = torch.optim.AdamW(groups, lr=3e-4, betas=(0.9, 0.95))

    assert any(g["lr_scale"] == 25.0 for g in opt.param_groups), "no boosted group present"
    assert any(g["lr_scale"] == 1.0 for g in opt.param_groups), "no base group present"

    for step, base_lr in enumerate([1e-5, 3e-4, 1.5e-4, 3e-5]):
        for g in opt.param_groups:
            g["lr"] = base_lr * g.get("lr_scale", 1.0)
        for g in opt.param_groups:
            expected = base_lr * g["lr_scale"]
            assert abs(g["lr"] - expected) < 1e-12, (
                f"step {step}: lr {g['lr']} != base_lr*scale {expected} "
                f"(lr_scale was likely overwritten, not multiplied)")


def test_default_scale_is_inert():
    """ssm_lr_scale=1.0 (the default) must be bit-identical to no scaling at all."""
    torch.manual_seed(0)
    m = build_model("mamba", vocab_size=32, n_layers=2, d_model=32, d_state=8,
                    d_conv=4, expand=2, dropout=0.0)
    groups = build_param_groups(m, weight_decay=0.1, ssm_lr_scale=1.0)
    assert all(g["lr_scale"] == 1.0 for g in groups)
