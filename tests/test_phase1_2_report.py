"""Gate for one plan2.md 1.2 setting (scripts/phase1_2_report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase1_2_report import setting_gate


def rec(acc, complete=True):
    return {"complete": complete, "history": [{"step": 3000, "val_acc": acc}]}


def test_setting_gate():
    s = [rec(0.99), rec(0.98), rec(0.97)]
    assert setting_gate(s, [rec(0.02)] * 3)["verdict"] == "PASS"
    assert setting_gate([rec(0.99), rec(0.9), rec(0.99)], [rec(0.02)] * 3)["verdict"] == "FAIL"
    assert setting_gate(s, [rec(0.6)] * 3)["verdict"] == "FAIL"
    assert setting_gate(s, [rec(0.02)] * 2)["verdict"] == "INCOMPLETE"
    assert setting_gate(s, [rec(0.02), rec(0.02), rec(0.02, complete=False)])["verdict"] == "INCOMPLETE"
