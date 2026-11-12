"""The custom and official-kernel Mamba must differ only in the mixer.

plan2.md 0.1: same group membership in the optimizer under every option, same
trainable parameter count (excluding the custom model's ablation-only fixed_*
params), same model-level init.

Every test runs twice: against the official mixer when mamba-ssm is importable (on
the Kaggle GPU), and always against tests/vendor/mamba_simple_init.py, a verbatim
copy of the official mixer's __init__ (names, shapes, init and tags), so the suite
also guards this on machines without CUDA.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from src.models import build_model, count_parameters
from src.models.mamba_official import OracleMamba
from src.optim import build_param_groups, is_no_decay, is_ssm_selectivity
from vendor.mamba_simple_init import Mamba as VendoredMixer

# configs/recall/mamba_small.yaml
SMALL = dict(n_layers=4, d_model=124, d_state=16, d_conv=4, expand=2, dropout=0.0)
VOCAB = 64


def _real_mixer():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except Exception:
        return None
    return Mamba


MIXERS = [pytest.param(VendoredMixer, id="vendored")]
if _real_mixer() is not None:
    MIXERS.append(pytest.param(_real_mixer(), id="mamba_ssm"))


def _pair(mixer_cls, **init):
    torch.manual_seed(0)
    custom = build_model("mamba", vocab_size=VOCAB, **SMALL, **init)
    torch.manual_seed(0)
    official = OracleMamba(vocab_size=VOCAB, **SMALL, mixer_cls=mixer_cls, **init)
    return custom, official


def _shared_params(model):
    """name -> param, with the official naming mapped onto the custom one and the
    custom model's ablation-only params dropped."""
    return {n.replace(".mixer.", ".ssm."): p for n, p in model.named_parameters()
            if p.requires_grad and ".fixed_" not in n}


@pytest.mark.parametrize("mixer_cls", MIXERS)
def test_same_parameter_names_and_shapes(mixer_cls):
    custom, official = _pair(mixer_cls)
    c, o = _shared_params(custom), _shared_params(official)
    assert sorted(c) == sorted(o)
    for n in c:
        assert c[n].shape == o[n].shape, n


@pytest.mark.parametrize("mixer_cls", MIXERS)
def test_equal_trainable_parameter_count(mixer_cls):
    custom, official = _pair(mixer_cls)

    def fixed(m):
        return sum(p.numel() for n, p in m.named_parameters() if ".fixed_" in n)
    assert count_parameters(custom) == count_parameters(official)
    # The numbers the configs quote, so a stale comment is caught next time.
    assert count_parameters(official) - fixed(official) == 448_632
    assert fixed(custom) == fixed(official) == 1_120


@pytest.mark.parametrize("mixer_cls", MIXERS)
@pytest.mark.parametrize("decay_1d", [False, True])
@pytest.mark.parametrize("ssm_lr_scale", [1.0, 10.0])
def test_identical_group_membership(mixer_cls, decay_1d, ssm_lr_scale):
    custom, official = _pair(mixer_cls)

    def membership(model):
        shared = _shared_params(model)
        by_id = {id(p): n for n, p in shared.items()}
        out = {}
        for g in build_param_groups(model, 0.1, ssm_lr_scale=ssm_lr_scale,
                                    decay_1d=decay_1d):
            for p in g["params"]:
                if id(p) in by_id:
                    out[by_id[id(p)]] = (g["weight_decay"], g["lr_scale"])
        assert set(out) == set(shared)
        return out

    assert membership(custom) == membership(official)


@pytest.mark.parametrize("mixer_cls", MIXERS)
def test_decay_1d_moves_only_biases_and_norm_gains(mixer_cls):
    _, official = _pair(mixer_cls)
    for n, p in official.named_parameters():
        off, on = is_no_decay(n, p, False), is_no_decay(n, p, True)
        if n.endswith("A_log") or n.endswith(".D"):
            assert off and on, n                      # dynamics: never decayed
        elif p.ndim < 2:
            assert off and not on, n                  # biases, norm gains
        else:
            assert not off and not on, n              # matrices: always decayed


def test_decay_1d_keeps_tcn_weight_g_exempt():
    m = build_model("tcn", vocab_size=32, n_layers=3, d_model=32, kernel_size=3,
                    dropout=0.0)
    gs = [(n, p) for n, p in m.named_parameters() if "weight_g" in n]
    assert gs
    assert all(is_no_decay(n, p, decay_1d=True) for n, p in gs)


@pytest.mark.parametrize("mixer_cls", MIXERS)
@pytest.mark.parametrize("emb_init_std", [0.02, 1.0])
@pytest.mark.parametrize("depth_scale", [False, True])
def test_model_level_init_matches(mixer_cls, emb_init_std, depth_scale):
    custom, official = _pair(mixer_cls, emb_init_std=emb_init_std,
                             out_proj_depth_scale=depth_scale)
    for m in (custom, official):
        assert m.head.weight is m.tok_emb.weight
        assert m.tok_emb.weight.std().item() == pytest.approx(emb_init_std, rel=0.05)

    # kaiming_uniform(a=sqrt(5)) on fan_in d_inner: bound 1/sqrt(d_inner).
    d_inner = SMALL["d_model"] * SMALL["expand"]
    bound = 1 / math.sqrt(d_inner) / (math.sqrt(SMALL["n_layers"]) if depth_scale else 1)
    expected_std = bound / math.sqrt(3)
    c, o = _shared_params(custom), _shared_params(official)
    for n in c:
        if n.endswith("out_proj.weight"):
            for p in (c[n], o[n]):
                assert p.abs().max().item() <= bound + 1e-6, n
                assert p.std().item() == pytest.approx(expected_std, rel=0.05), n
        if n.endswith("dt_proj.bias"):
            # _no_reinit: survives model-level init in both models
            for p in (c[n], o[n]):
                dt = torch.nn.functional.softplus(p)
                assert dt.min() >= 1e-4 and dt.max() <= 0.1 + 1e-6, n


def test_defaults_are_the_reference_recipe():
    custom, official = _pair(VendoredMixer)
    for m in (custom, official):
        assert m.emb_init_std == 0.02
        assert m.out_proj_depth_scale is True


def test_mamba_ssm_registered_and_lazy():
    """build_model('mamba_ssm') is registered; without mamba-ssm it fails with an
    ImportError at construction, not at `import src.models`."""
    if _real_mixer() is None:
        with pytest.raises(ImportError):
            build_model("mamba_ssm", vocab_size=VOCAB, **SMALL)
    else:
        m = build_model("mamba_ssm", vocab_size=VOCAB, **SMALL)
        assert count_parameters(m) == 448_632


@pytest.mark.skipif(_real_mixer() is None or not torch.cuda.is_available(),
                    reason="needs mamba-ssm and CUDA")
def test_official_forward_shape():
    m = build_model("mamba_ssm", vocab_size=VOCAB, **SMALL).cuda()
    x = torch.randint(0, VOCAB, (2, 64), device="cuda")
    assert m(x).shape == (2, 64, VOCAB)


@pytest.mark.parametrize("mixer_cls", MIXERS)
def test_copy_official_to_custom_makes_parameters_identical(mixer_cls):
    from src.models.mamba_official import copy_official_to_custom
    torch.manual_seed(0)
    official = OracleMamba(vocab_size=VOCAB, **SMALL, mixer_cls=mixer_cls,
                           emb_init_std=1.0)
    torch.manual_seed(1)
    custom = build_model("mamba", vocab_size=VOCAB, **SMALL)
    written = copy_official_to_custom(official, custom)
    c, o = _shared_params(custom), _shared_params(official)
    assert sorted(w for w in written if ".fixed_" not in w) == sorted(c)
    for n in c:
        assert torch.equal(c[n], o[n]), n
    # tied head follows the copied embedding
    assert torch.equal(custom.head.weight, official.tok_emb.weight)


@pytest.mark.skipif(_real_mixer() is None or not torch.cuda.is_available(),
                    reason="needs mamba-ssm and CUDA")
def test_copied_models_agree_on_forward():
    from src.models.mamba_official import copy_official_to_custom
    torch.manual_seed(0)
    official = build_model("mamba_ssm", vocab_size=VOCAB, **SMALL).cuda()
    custom = build_model("mamba", vocab_size=VOCAB, **SMALL).cuda()
    copy_official_to_custom(official, custom)
    official.eval(); custom.eval()
    x = torch.randint(0, VOCAB, (2, 64), device="cuda")
    with torch.no_grad():
        assert (official(x) - custom(x)).abs().max().item() < 1e-3


# ── 1.2a: the non-selective ablation on the official model ────────────────────
from vendor.selective_scan_ref import selective_scan_ref


def _ablated_pair(seed=0, real=False):
    """Custom model and an official model holding the custom weights (including the
    ablation constants), both evaluated with input_independent=True."""
    torch.manual_seed(seed)
    custom = build_model("mamba", vocab_size=VOCAB, **SMALL, emb_init_std=1.0)
    if real:
        official = build_model("mamba_ssm", vocab_size=VOCAB, **SMALL, use_fast_path=False)
    else:
        official = OracleMamba(vocab_size=VOCAB, **SMALL, mixer_cls=VendoredMixer,
                               scan_fn=selective_scan_ref)
    from src.models.mamba_official import copy_custom_to_official
    # train a few random steps first so the constants are not at init
    with torch.no_grad():
        for n, p in custom.named_parameters():
            if ".fixed_" in n:
                p.add_(0.05 * torch.randn_like(p))
    copy_custom_to_official(custom, official)
    custom.eval(); official.eval()
    return custom, official


def _check_ablated_match(custom, official, device):
    custom, official = custom.to(device), official.to(device)
    x = torch.randint(0, VOCAB, (2, 64), device=device)
    with torch.no_grad():
        a = custom(x, input_independent=True)
        b = official(x, input_independent=True)
    assert (a - b).abs().max().item() < 1e-4


def _check_ignores_selectivity_weights(official, device):
    official = official.to(device)
    x = torch.randint(0, VOCAB, (2, 64), device=device)
    with torch.no_grad():
        before = official(x, input_independent=True)
        for b in official.blocks:
            b.mixer.x_proj.weight.add_(torch.randn_like(b.mixer.x_proj.weight))
            b.mixer.dt_proj.weight.add_(torch.randn_like(b.mixer.dt_proj.weight))
            b.mixer.dt_proj.bias.add_(torch.randn_like(b.mixer.dt_proj.bias))
        after = official(x, input_independent=True)
    assert torch.equal(before, after)


def test_official_has_ablation_params_and_rest_of_init_unchanged(monkeypatch):
    """The constants are created last, so at a given seed every other tensor is
    exactly what it was before 1.2a (rebuilt here with the constants switched off)."""
    import src.models.mamba_official as mo
    torch.manual_seed(0)
    with_c = OracleMamba(vocab_size=VOCAB, **SMALL, mixer_cls=VendoredMixer)
    assert sum(".fixed_" in n for n, _ in with_c.named_parameters()) == 3 * SMALL["n_layers"]
    monkeypatch.setattr(mo, "add_ablation_params", lambda mixer: None)
    torch.manual_seed(0)
    without = OracleMamba(vocab_size=VOCAB, **SMALL, mixer_cls=VendoredMixer)
    ref = dict(without.named_parameters())
    for n, p in with_c.named_parameters():
        if ".fixed_" not in n:
            assert torch.equal(p, ref[n]), n


def test_official_ablation_matches_custom_ablation_cpu():
    custom, official = _ablated_pair()
    _check_ablated_match(custom, official, "cpu")


def test_official_ablation_ignores_x_proj_and_dt_proj_cpu():
    _, official = _ablated_pair()
    _check_ignores_selectivity_weights(official, "cpu")


def test_official_selective_path_unchanged_by_ablation_flag_default():
    _, official = _ablated_pair()
    # selective forward with the vendored mixer is not runnable (no kernels); the
    # default flag must route to the mixer, which raises NotImplementedError
    with pytest.raises(NotImplementedError):
        official(torch.randint(0, VOCAB, (1, 8)))


@pytest.mark.skipif(_real_mixer() is None or not torch.cuda.is_available(),
                    reason="needs mamba-ssm and CUDA")
def test_official_ablation_matches_custom_ablation_cuda():
    custom, official = _ablated_pair(real=True)
    _check_ablated_match(custom, official, "cuda")


@pytest.mark.skipif(_real_mixer() is None or not torch.cuda.is_available(),
                    reason="needs mamba-ssm and CUDA")
def test_official_ablation_ignores_x_proj_and_dt_proj_cuda():
    _, official = _ablated_pair(real=True)
    _check_ignores_selectivity_weights(official, "cuda")
