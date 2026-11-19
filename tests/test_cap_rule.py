"""The D3 CAP-extension rule (scripts/cap_rule.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from cap_rule import needs_extension


def rec(accs, stopped_early=False):
    return {"stopped_early": stopped_early,
            "history": [{"step": 250 * (i + 1), "val_acc": a} for i, a in enumerate(accs)]}


def test_flat_low_is_final():
    assert needs_extension(rec([0.02] * 16)) == []


def test_high_is_final_and_early_stop_never_extends():
    assert needs_extension(rec([0.5] * 15 + [0.95])) == [] or \
        all(r.startswith("unconverged") for r in needs_extension(rec([0.5] * 15 + [0.95])))
    assert needs_extension(rec([0.5] * 16, stopped_early=True)) == []


def test_ambiguous_extends():
    assert any(r.startswith("ambiguous") for r in needs_extension(rec([0.5] * 16)))


def test_rising_extends_even_below_ambiguous_band():
    accs = [0.02] * 12 + [0.05, 0.08, 0.12, 0.15]     # +0.13 over the last 1000 steps
    assert any(r.startswith("unconverged") for r in needs_extension(rec(accs)))
