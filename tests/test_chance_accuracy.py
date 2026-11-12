"""chance_accuracy matches each generator's answer alphabet (plan2.md decision
2026-11-12: selective_copy chance is 1/62 at vocab 64, not 1/31)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks import chance_accuracy, make_batcher, record_chance


def test_selective_copy_alphabet():
    b = make_batcher("selective_copy", seq_len=64, n_data=4, vocab_size=64)
    _, tgt, mask, _ = b(256, 0)
    seen = set(tgt[mask].tolist())
    assert seen <= set(range(2, 64)) and len(seen) == 62
    assert chance_accuracy("selective_copy", 64) == 1 / 62


def test_mqar_alphabet():
    b = make_batcher("mqar", seq_len=64, n_pairs=4, vocab_size=64)
    _, tgt, mask, _ = b(512, 0)
    seen = set(tgt[mask].tolist())
    assert len(seen) == 31
    assert chance_accuracy("mqar", 64) == 1 / 31


def test_record_chance_ignores_stored_value():
    old = {"task": "selective_copy", "task_args": {"vocab_size": 64}, "chance_acc": 1 / 31}
    assert record_chance(old) == 1 / 62
