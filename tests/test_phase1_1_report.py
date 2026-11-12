"""CAP, projection and gate logic for plan2.md 1.1 (scripts/phase1_1_report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase1_1_report import cap_from_pilot, gate_c, project_gpu_min, steps_to


def rec(accs):
    return {"history": [{"step": 250 * (i + 1), "val_acc": a} for i, a in enumerate(accs)]}


def test_steps_to_first_eval_at_or_above_target():
    assert steps_to(rec([0.1, 0.94, 0.95, 0.99])) == 750
    assert steps_to(rec([0.1, 0.5])) is None


def test_cap_rounds_up_and_is_capped():
    assert cap_from_pilot([1250, 1500, 1750]) == 3000      # 1.5 x 1750 = 2625 -> 3000
    assert cap_from_pilot([2000, 2000, 2000]) == 3000      # exactly 3000 stays 3000
    assert cap_from_pilot([14000, 1000, 1000]) == 20000    # 21000 -> capped
    assert cap_from_pilot([1000, None, 1000]) is None      # a seed never reached 0.95


def test_projection_waves():
    # 6 runs, 3 at a time = 2 waves of 3000 steps at 1 step/s = 100 min, +5%, +3 setup
    assert abs(project_gpu_min(3000, 6, 3, 1.0) - (3 + 2 * 50 * 1.05)) < 1e-9
    assert abs(project_gpu_min(3000, 6, 2, 1.0) - (3 + 3 * 50 * 1.05)) < 1e-9


def test_gate_c():
    s = [rec([0.99]), rec([0.97]), rec([0.96])]
    a = [rec([0.05]), rec([0.04]), rec([0.06])]
    assert gate_c(s, a)["verdict"] == "PASS"
    s_bad = [rec([0.99]), rec([0.94]), rec([0.99])]
    assert gate_c(s_bad, a)["verdict"] == "FAIL"
    a_close = [rec([0.6]), rec([0.6]), rec([0.6])]
    assert gate_c(s, a_close)["verdict"] == "FAIL"
    assert gate_c(s[:2], a)["verdict"] == "INCOMPLETE"
