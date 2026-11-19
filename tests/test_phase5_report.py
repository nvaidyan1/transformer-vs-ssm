"""Gates for plan2.md 5.2 and 5.3 (scripts/phase5_report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase5_report import gate_52, gate_53


def rec(mse):
    return {"final_val_mse": mse}


def cells(stat_sel, stat_abl, sw_sel, sw_abl):
    return {("stationary", "mamba", 1): {s: rec(stat_sel) for s in range(3)},
            ("stationary", "mamba_ablated", 1): {s: rec(stat_abl) for s in range(3)},
            ("switching", "mamba", 1): {s: rec(sw_sel) for s in range(3)},
            ("switching", "mamba_ablated", 1): {s: rec(v) for s, v in enumerate(sw_abl)}}


def test_gate_52():
    assert gate_52(cells(0.10, 0.105, 0.10, [0.12, 0.13, 0.20]))["verdict"] == "PASS"
    assert gate_52(cells(0.10, 0.12, 0.10, [0.12, 0.13, 0.20]))["verdict"] == "FAIL"   # (a)
    assert gate_52(cells(0.10, 0.10, 0.10, [0.12, 0.11, 0.20]))["verdict"] == "FAIL"   # (b) one seed
    assert gate_52({})["verdict"] == "INCOMPLETE"


def test_gate_53():
    g = {"models": {f"switching_mamba_L1_s{s}": {"arch": "mamba", "n_layers": 1, "pearson_r": r}
                    for s, r in enumerate([0.8, 0.75, 0.9])}}
    assert gate_53(g)["verdict"] == "PASS"
    g["models"]["switching_mamba_L1_s1"]["pearson_r"] = 0.6
    assert gate_53(g)["verdict"] == "FAIL"
