"""Aggregation for plan2.md 1.1d (scripts/phase1_1d_report.py)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase1_1d_report import collect


def test_collect_excludes_partial_and_averages_slots(tmp_path):
    def w(name, acc, slots, complete=True):
        json.dump({"complete": complete, "slot_acc": slots,
                   "history": [{"step": 4000, "val_acc": acc}]}, open(tmp_path / name, "w"))
    w("ablated_nd2_s0.json", 0.6, [0.2, 1.0])
    w("ablated_nd2_s1.json", 0.4, [0.0, 0.8])
    w("ablated_nd2_s2.json", 0.9, [0.8, 1.0], complete=False)
    s = collect(tmp_path)[("ablated", 2)]
    assert s["n"] == 2 and abs(s["mean"] - 0.5) < 1e-12
    assert s["slot_acc_mean"] == [0.1, 0.9]
    assert collect(tmp_path)[("selective", 1)]["n"] == 0
