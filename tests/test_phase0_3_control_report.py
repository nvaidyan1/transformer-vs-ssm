"""Gate for the 0.3 Transformer control (scripts/phase0_3_control_report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase0_3_control_report import summarize


def rec(seed, acc, step=4000, complete=True):
    return {"seed": seed, "lr": 1e-3, "n_params": 452_296, "complete": complete,
            "history": [{"step": step, "val_acc": acc}]}


def test_pass_on_mean():
    assert summarize([rec(0, 0.99), rec(1, 0.93), rec(2, 0.99)])["gate"] == "PASS"


def test_fail_on_mean():
    assert summarize([rec(0, 0.96), rec(1, 0.90), rec(2, 0.95)])["gate"] == "FAIL"


def test_incomplete_seed_blocks_gate():
    rs = [rec(0, 1.0), rec(1, 1.0), rec(2, 1.0, step=3000)]
    assert summarize(rs)["gate"] == "INCOMPLETE"
