"""Gate and histogram summary for plan2.md 1.1b (scripts/phase1_1b_report.py)."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase1_1b_report import gate, hist_stats


def rec(acc, complete=True):
    return {"complete": complete, "history": [{"step": 3000, "val_acc": acc}]}


def test_gate():
    assert gate(rec(0.95)) == "PASS"
    assert gate(rec(0.9)) == "PASS"
    assert gate(rec(0.89)) == "FAIL"
    assert gate(rec(0.99, complete=False)) == "INCOMPLETE"
    assert gate(None) == "INCOMPLETE"


def test_hist_stats_uniform():
    h = {"pred": {str(k): 10 for k in range(2, 64)}, "target": {str(k): 10 for k in range(2, 64)}}
    st = hist_stats(h)
    assert st["distinct_predicted"] == 62 and st["non_data_predictions"] == 0
    assert abs(st["pred_entropy_nats"] - math.log(62)) < 1e-9
