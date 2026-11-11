"""scripts/phase0_weight_copy.py: lockstep training on identical batches.

Two weight-identical custom models trained by train_pair must stay identical on CPU
(deterministic), so any divergence in the real run comes from the models, not from
the harness feeding them differently."""

import copy
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from phase0_weight_copy import train_pair
from src.models import build_model
from src.tasks import make_batcher


def test_identical_models_stay_identical():
    cfg = yaml.safe_load(open(ROOT / "configs/recall/mamba_small.yaml"))
    cfg.update(eval_every=2, eval_batches=1, batch_size=4, warmup_steps=1)
    torch.manual_seed(0)
    a = build_model("mamba", vocab_size=64, n_layers=1, d_model=16, d_state=4,
                    d_conv=4, expand=2, dropout=0.0)
    b = copy.deepcopy(a)
    batcher = make_batcher("selective_copy", n_data=4, seq_len=32, vocab_size=64)
    hist = train_pair({"a": a, "b": b}, cfg, batcher, torch.device("cpu"),
                      seed=0, steps=4, decay_1d=False)
    assert [h["step"] for h in hist] == [2, 4]
    for h in hist:
        assert h["a"] == h["b"]
    for (n, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(p, q), n
    # and training actually moved the weights
    fresh = build_model("mamba", vocab_size=64, n_layers=1, d_model=16, d_state=4,
                        d_conv=4, expand=2, dropout=0.0)
    assert not torch.equal(a.tok_emb.weight, fresh.tok_emb.weight)


def test_report_rows_paired_difference():
    from phase0_weight_copy_report import rows, to_markdown
    rec = {"emb_init_std": 1.0, "decay_1d": False, "seed": 0, "complete": True,
           "init_max_logit_diff": 1e-6, "chance_acc": 1 / 31,
           "history": [{"step": 250, "official": {"val_acc": 0.2}, "custom": {"val_acc": 0.1}}]}
    assert abs(rows(rec)[0]["diff"] - (-0.1)) < 1e-12
    assert "| 250 | 0.2000 | 0.1000 | -0.1000 |" in to_markdown(rec)
