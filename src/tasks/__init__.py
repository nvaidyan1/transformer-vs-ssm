"""Synthetic diagnostic tasks.

mqar           — Multi-Query Associative Recall (canonical; the experimental spine)
selective_copy — Selective Copying (canonical; the Mamba implementation gate)

Both expose the same contract, so the trainer never special-cases a task:

    batcher = make_batcher("mqar", seq_len=128, n_pairs=4, vocab_size=64)
    tokens, targets, loss_mask, metas = batcher(batch_size=32, seed=123)

`targets` is IGNORE_INDEX everywhere except the positions that carry the answer, so
`cross_entropy(..., ignore_index=IGNORE_INDEX)` masks the objective for free.
"""

from functools import partial

from src.tasks.mqar import IGNORE_INDEX, MQARSample, generate_mqar
from src.tasks.mqar import make_batch as make_mqar_batch
from src.tasks.selective_copy import SelectiveCopySample, generate_selective_copy
from src.tasks.selective_copy import make_batch as make_selective_copy_batch

TASKS = ("mqar", "selective_copy")


def make_batcher(task: str, **task_kwargs):
    """Return `fn(batch_size, seed) -> (tokens, targets, loss_mask, metas)`.

    Args:
        task:         one of TASKS
        **task_kwargs: task-specific shape parameters, e.g.
                       mqar           -> seq_len, n_pairs, vocab_size, n_queries
                       selective_copy -> seq_len, n_data, vocab_size
    """
    if task == "mqar":
        base = make_mqar_batch
    elif task == "selective_copy":
        base = make_selective_copy_batch
    else:
        raise ValueError(f"Unknown task {task!r}. Choose from {TASKS}")

    def batcher(batch_size: int, seed: int):
        return base(batch_size, seed=seed, **task_kwargs)

    return batcher


__all__ = [
    "TASKS", "make_batcher", "IGNORE_INDEX",
    "generate_mqar", "make_mqar_batch", "MQARSample",
    "generate_selective_copy", "make_selective_copy_batch", "SelectiveCopySample",
]


def chance_accuracy(task: str, vocab_size: int, **_ignored) -> float:
    """Accuracy of uniform guessing over the answer symbols.

    mqar:           values are drawn from (vocab_size - 1) // 2 ids -> 1 / that.
    selective_copy: data are drawn from ids [2, vocab_size) -> 1 / (vocab_size - 2).
    Before 2026-11-12 the trainer used the MQAR formula for both tasks, recording 1/31
    instead of 1/62 for selective_copy at vocab 64 (results/LOG.md).
    """
    if task == "mqar":
        return 1.0 / max(1, (vocab_size - 1) // 2)
    if task == "selective_copy":
        return 1.0 / (vocab_size - 2)
    raise ValueError(f"Unknown task {task!r}")


def record_chance(record: dict) -> float:
    """chance_accuracy for a saved result record (ignores its stored chance_acc, which
    is wrong for selective_copy records written before 2026-11-12)."""
    args = record.get("task_args", {})
    return chance_accuracy(record.get("task", "selective_copy"),
                           args.get("vocab_size", 64))
