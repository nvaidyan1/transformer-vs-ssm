"""configs/recall/*_small.yaml: parameter matching and the locked mamba_ssm recipe
(plan2.md 0.3). mamba_ssm is counted via the vendored official constructor, so this
runs without CUDA."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.models import build_model, count_parameters
from src.models.mamba_official import OracleMamba
from vendor.mamba_simple_init import Mamba as VendoredMixer

CFG = ROOT / "configs/recall"


def load(name):
    return yaml.safe_load(open(CFG / f"{name}.yaml"))


def n_params(name):
    c = load(name)
    if c["arch"] == "mamba_ssm":
        m = OracleMamba(vocab_size=c["vocab_size"], mixer_cls=VendoredMixer, **c["model"])
    else:
        m = build_model(c["arch"], vocab_size=c["vocab_size"], **c["model"])
    fixed = sum(p.numel() for n, p in m.named_parameters() if ".fixed_" in n)
    return count_parameters(m) - fixed


def test_parameter_counts_matched_within_5_percent():
    ref = n_params("mamba_ssm_small")
    assert ref == 448_632
    assert n_params("mamba_small") == ref
    for name, expected in (("transformer_small", 452_296), ("tcn_small", 450_212)):
        n = n_params(name)
        assert n == expected, name            # the number quoted in the config
        assert abs(n / ref - 1) <= 0.05, name


def test_mamba_ssm_recipe_locked():
    c = load("mamba_ssm_small")
    assert c["arch"] == "mamba_ssm"
    assert c["model"]["emb_init_std"] == 1.0
    assert c["model"]["out_proj_depth_scale"] is True
    assert c["decay_1d"] is False


def test_custom_mamba_keeps_reference_init():
    m = load("mamba_small")["model"]
    assert m.get("emb_init_std", 0.02) == 0.02


def test_shared_training_settings():
    names = ("transformer_small", "tcn_small", "mamba_small", "mamba_ssm_small")
    for key in ("vocab_size", "batch_size", "warmup_steps", "max_steps", "eval_every",
                "eval_batches", "weight_decay", "grad_clip"):
        assert len({load(n)[key] for n in names}) == 1, key


def test_learning_rates_locked_by_lr_check():
    """results/phase0_3_lr: picked by the pre-registered selection rule."""
    assert load("transformer_small")["lr"] == 1.0e-3
    assert load("tcn_small")["lr"] == 3.0e-4
    assert load("mamba_ssm_small")["lr"] == 1.0e-3
