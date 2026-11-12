"""Gate for plan2.md 1.1c, and the generator fact the report states."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase1_1c_report import gate
from src.tasks.selective_copy import MARKER_ID, NOISE_ID, make_batch


def rec(acc, complete=True):
    return {"complete": complete, "history": [{"step": 3000, "val_acc": acc}]}


def test_gate():
    assert gate(rec(0.9)) == "PASS"
    assert gate(rec(0.89)) == "FAIL"
    assert gate(rec(0.99, complete=False)) == "INCOMPLETE"
    assert gate(None) == "INCOMPLETE"


def test_non_data_positions_are_one_blank_token():
    toks, _, _, metas = make_batch(16, 64, 4, 64, seed=0)
    for t, m in zip(toks, metas):
        inp = t[:m["input_len"]]
        non_data = np.delete(inp, m["source_positions"])
        assert set(non_data.tolist()) == {NOISE_ID}
        assert t[m["marker_position"]] == MARKER_ID
