"""Gate for plan2.md 3.2 (scripts/phase3_report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase3_report import gate


def rec(acc):
    return {"complete": True, "history": [{"step": 4000, "val_acc": acc}]}


def cells(far_acc, near_acc):
    c = {("tcn_short", g): [rec(far_acc)] * 3 for g in (160, 256, 512)}
    c[("tcn_short", 32)] = [rec(near_acc)] * 3
    return c


def test_gate_pass_and_fail():
    assert gate(cells(0.02, 0.97))["verdict"] == "PASS"
    assert gate(cells(0.10, 0.97))["verdict"] == "FAIL"     # far gaps above 2x chance
    assert gate(cells(0.02, 0.85))["verdict"] == "FAIL"     # near gap not solved


def test_gate_incomplete_without_all_seeds():
    c = cells(0.02, 0.97)
    c[("tcn_short", 256)] = [rec(0.02)] * 2
    assert gate(c)["verdict"] == "INCOMPLETE"
