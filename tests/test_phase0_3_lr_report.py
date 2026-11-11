"""Instability and selection rules for the 0.3 lr check (scripts/phase0_3_lr_report.py),
exactly as fixed in results/phase0_3_lr/PREDICTION.md."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase0_3_lr_report import instability, pick


def rec(lr, accs, vlosses=None, complete=True):
    steps = range(250, 250 * len(accs) + 1, 250)
    vlosses = vlosses or [3.0 - 0.1 * i for i in range(len(accs))]
    return {"arch": "x", "lr": lr, "complete": complete, "history": [
        {"step": s, "train_loss": v, "val_loss": v, "val_acc": a}
        for s, a, v in zip(steps, accs, vlosses)]}


FULL = 16   # 4000 / 250 evals


def test_stable_run():
    assert instability(rec(1e-3, [0.1 + 0.05 * i for i in range(FULL)])) == []


def test_nan_is_unstable():
    r = rec(1e-3, [0.5] * FULL)
    r["history"][3]["train_loss"] = float("nan")
    assert "non-finite loss" in instability(r)


def test_collapse_after_peak_is_unstable():
    accs = [0.9] * (FULL - 1) + [0.8]
    assert any(s.startswith("collapse") for s in instability(rec(3e-3, accs)))


def test_rising_val_loss_is_unstable():
    v = [2.0] * (FULL - 1) + [2.5]
    assert any(s.startswith("val_loss rose") for s in instability(rec(3e-3, [0.3] * FULL, v)))


def test_incomplete_not_selectable():
    assert instability(rec(1e-3, [0.5] * 8)) == ["incomplete"]
    assert instability(rec(1e-3, [0.5] * FULL, complete=False)) == ["incomplete"]


def test_pick_highest_stable_and_tie_goes_to_lower_lr():
    a = rec(3e-4, [0.90] * FULL)
    b = rec(1e-3, [0.990] * FULL)
    c = rec(3e-3, [0.993] * FULL)          # within 0.005 of best -> lower lr wins
    assert pick([a, b, c]) == 1e-3
    unstable_best = rec(3e-3, [1.0] * (FULL - 1) + [0.9])   # collapse
    assert pick([a, unstable_best]) == 3e-4
    assert pick([unstable_best]) is None
