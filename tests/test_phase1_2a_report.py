"""Gate for plan2.md 1.2a (scripts/phase1_2a_report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase1_2a_report import gate


def rec(acc, complete=True):
    return {"complete": complete, "history": [{"step": 3000, "val_acc": acc}]}


def test_gate():
    assert gate(rec(0.99), rec(0.02)) == "PASS"
    assert gate(rec(0.94), rec(0.02)) == "FAIL"
    assert gate(rec(0.99), rec(0.2)) == "FAIL"
    assert gate(rec(0.99), None) == "INCOMPLETE"
