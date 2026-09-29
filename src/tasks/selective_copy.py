"""
Selective Copying (Gu & Dao 2023, §4.1; after Jing et al. 2019).

Data tokens are scattered at RANDOM positions through a field of noise tokens. After a
marker, the model must reproduce the data tokens in their original order, ignoring the
noise.

    n7 D3 n2 n9 D1 n4 n1 D8 n5  |MARK|  D3 D1 D8
    └──────── input region ────┘         └ loss ┘

Why the randomisation matters
-----------------------------
In *vanilla* copying the data sits at fixed offsets, so a time-invariant model can solve
it with a fixed set of taps. Randomising the spacing removes that solution: the model
must decide *from content* which tokens to retain. That is exactly the capability
selectivity was introduced to provide, which is why this task is the right gate on a
Mamba implementation.

Use as an implementation check
------------------------------
This repo runs a bespoke pure-PyTorch sequential-scan Mamba (`src/models/mamba.py`), not
`mamba-ssm`. A poor MQAR result from it is uninterpretable until it demonstrates the
canonical capability here. Recommended control: train the same model with selection
disabled (`input_independent=True` in `MambaSSM.forward`, which freezes delta/B/C to
their sequence means, approximating a time-invariant SSM). A working implementation
should show selective >> input-independent. If the two match, selection is not doing
anything and the bug is in the model, not the story.

Token layout for a vocabulary of size V:
    id 0                  noise / blank
    id 1                  marker (start of output region)
    [2, V)                data symbols
"""

from dataclasses import dataclass, field

import numpy as np

IGNORE_INDEX = -100

NOISE_ID = 0
MARKER_ID = 1
DATA_LO = 2


@dataclass
class SelectiveCopySample:
    tokens: np.ndarray      # (T,) int64 — input, answers teacher-forced
    targets: np.ndarray     # (T,) int64 — IGNORE_INDEX except in the output region
    loss_mask: np.ndarray   # (T,) bool
    meta: dict = field(default_factory=dict)


def min_seq_len(n_data: int) -> int:
    """Shortest sequence: n_data data tokens + marker + n_data outputs."""
    return 2 * n_data + 1


def generate_selective_copy(
    seq_len: int,
    n_data: int,
    vocab_size: int,
    seed: int = None,
    rng: np.random.Generator = None,
) -> SelectiveCopySample:
    """Generate one Selective Copying example.

    Args:
        seq_len:    total sequence length (input region + marker + output region)
        n_data:     number of data tokens to memorise and reproduce
        vocab_size: total vocabulary; data symbols occupy [2, vocab_size)
        seed:       seed for a fresh generator (ignored if `rng` is given)
        rng:        an existing numpy Generator, for batch generation
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if vocab_size <= DATA_LO:
        raise ValueError(f"vocab_size must exceed {DATA_LO}, got {vocab_size}")
    need = min_seq_len(n_data)
    if seq_len < need:
        raise ValueError(f"seq_len={seq_len} too short for n_data={n_data} (need >= {need})")

    input_len = seq_len - 1 - n_data          # everything before the marker
    if input_len < n_data:
        raise ValueError(f"input region {input_len} cannot hold {n_data} data tokens")

    # Data symbols and their RANDOM positions within the input region.
    data = rng.integers(DATA_LO, vocab_size, size=n_data)
    positions = np.sort(rng.choice(input_len, size=n_data, replace=False))

    tokens = np.full(seq_len, NOISE_ID, dtype=np.int64)
    tokens[positions] = data

    marker_pos = input_len
    tokens[marker_pos] = MARKER_ID
    tokens[marker_pos + 1:] = data            # teacher-forced output region

    targets = np.full(seq_len, IGNORE_INDEX, dtype=np.int64)
    loss_mask = np.zeros(seq_len, dtype=bool)
    # Predict data[i] from position marker_pos + i.
    for i in range(n_data):
        targets[marker_pos + i] = data[i]
        loss_mask[marker_pos + i] = True

    gaps = np.diff(positions) if n_data > 1 else np.array([], dtype=np.int64)
    return SelectiveCopySample(
        tokens=tokens,
        targets=targets,
        loss_mask=loss_mask,
        meta={
            "data": data.astype(int).tolist(),
            "source_positions": positions.astype(int).tolist(),
            "gaps": gaps.astype(int).tolist(),
            "marker_position": int(marker_pos),
            "input_len": int(input_len),
            "n_data": int(n_data),
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
        },
    )


def make_batch(
    batch_size: int,
    seq_len: int,
    n_data: int,
    vocab_size: int,
    seed: int = None,
):
    """Stack `batch_size` independent Selective Copying examples."""
    rng = np.random.default_rng(seed)
    samples = [generate_selective_copy(seq_len, n_data, vocab_size, rng=rng)
               for _ in range(batch_size)]
    return (
        np.stack([s.tokens for s in samples]),
        np.stack([s.targets for s in samples]),
        np.stack([s.loss_mask for s in samples]),
        [s.meta for s in samples],
    )
