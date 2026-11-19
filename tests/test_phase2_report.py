"""Slope fitting and gates for plan2.md Phase 2 (scripts/phase2_report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase2_report import gate_21, gate_22, slope, slopes


def test_slope_recovers_power_law():
    xs = [128, 256, 512, 1024]
    assert abs(slope(xs, [x ** 2 for x in xs]) - 2.0) < 1e-9
    assert abs(slope(xs, [3 * x for x in xs]) - 1.0) < 1e-9


def test_slopes_skip_oom_rows_and_use_top_three():
    rows = [{"length": L, "activation_bytes": L ** 2, "time_s": L} for L in (128, 256, 512, 1024)]
    rows.append({"length": 2048, "activation_bytes": None, "time_s": None})
    s = slopes(rows)
    assert s["oom_lengths"] == [2048] and s["top3_lengths"] == [256, 512, 1024]
    assert abs(s["memory_top3"] - 2) < 1e-9 and abs(s["time_top3"] - 1) < 1e-9


def test_gate_21():
    good = {"transformer_naive": {"memory_top3": 1.9}, "transformer_sdpa": {"memory_top3": 1.1, "time_top3": 1.8},
            "tcn": {"memory_top3": 1.0, "time_top3": 0.9}, "mamba_ssm": {"memory_top3": 1.0, "time_top3": 1.2}}
    assert gate_21(good)["verdict"] == "PASS"
    bad = dict(good, tcn={"memory_top3": 1.0, "time_top3": 0.6})
    assert gate_21(bad)["verdict"] == "FAIL"


def test_gate_22():
    rec = {"archs": {"a": {"measured_bytes": [100, 200], "analytic_bytes": [100, 196]}}}
    assert gate_22(rec)["verdict"] == "PASS"
    rec["archs"]["a"]["analytic_bytes"] = [100, 150]
    assert gate_22(rec)["verdict"] == "FAIL"
