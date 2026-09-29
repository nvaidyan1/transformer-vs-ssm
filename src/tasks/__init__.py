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
