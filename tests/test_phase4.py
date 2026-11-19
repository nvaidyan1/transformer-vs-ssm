"""Phase 4 configs (parameter matching, locked lrs) and gate."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from phase4_report import gate
from src.models import build_model, count_parameters
from src.models.mamba_official import OracleMamba
from src.tasks import make_batcher
from vendor.mamba_simple_init import Mamba as VendoredMixer


def n_params(name):
    c = yaml.safe_load(open(ROOT / f"configs/mqar/{name}.yaml"))
    if c["arch"] == "mamba_ssm":
        m = OracleMamba(vocab_size=c["vocab_size"], mixer_cls=VendoredMixer, **c["model"])
    else:
        m = build_model(c["arch"], vocab_size=c["vocab_size"], **c["model"])
    return count_parameters(m) - sum(p.numel() for n, p in m.named_parameters() if ".fixed_" in n), c


def test_matched_within_5_percent_and_locked_lr():
    for s in (32, 64, 128):
        nm, cm = n_params(f"mamba_ssm_d{s}")
        nt, ct = n_params(f"transformer_m{s}")
        assert abs(nt / nm - 1) <= 0.05, s
        assert cm["lr"] == ct["lr"] == 1.0e-3
        assert ct["model"]["max_seq_len"] >= 4 * 64


def test_mqar_holds_64_pairs_at_vocab_256():
    b = make_batcher("mqar", seq_len=256, n_pairs=64, vocab_size=256)
    tok, tgt, mask, _ = b(4, 0)
    assert tok.shape == (4, 256) and mask.sum() == 4 * 64


def rec(acc):
    return {"complete": True, "history": [{"step": 4000, "val_acc": acc}]}


def test_gate():
    cells = {("mamba_ssm_d32", 4): [rec(0.99)] * 3, ("mamba_ssm_d32", 64): [rec(0.3)] * 3,
             ("transformer_m32", 4): [rec(0.99)] * 3, ("transformer_m32", 64): [rec(0.95)] * 3}
    assert gate(cells)["verdict"] == "PASS"
    cells[("mamba_ssm_d32", 64)] = [rec(0.8)] * 3
    assert gate(cells)["verdict"] == "FAIL"
    del cells[("transformer_m32", 64)]
    assert gate(cells)["verdict"] == "INCOMPLETE"


def test_gate_waits_for_extensions():
    cells = {("mamba_ssm_d32", 4): [rec(0.99)] * 3, ("mamba_ssm_d32", 64): [rec(0.3)] * 3,
             ("transformer_m32", 4): [rec(0.99)] * 3, ("transformer_m32", 64): [rec(0.95)] * 3}
    assert gate(cells, {("transformer_m32", 4)})["verdict"] == "INCOMPLETE"
    assert gate(cells, {("mamba_ssm_d64", 4)})["verdict"] == "PASS"
