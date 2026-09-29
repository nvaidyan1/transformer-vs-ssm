"""
Multi-Query Associative Recall (MQAR).

The task: key-value pairs are presented in a sequence, then several of those keys are
queried; for each query the model must produce the value bound to that key.

    ... K7 V7 ... K2 V2 ... K11 V11 ...    K2 V2    K11 V11
                                           ^loss    ^loss

Loss is computed ONLY at query positions (predicting the following value). Everything
else — the presentation phase and all filler — is masked out. This is deliberate: an
unmasked next-token objective over filler would recreate the enwik8 problem on a
synthetic task, letting positions irrelevant to retrieval dominate the loss.

Design notes
------------
- Keys, values, and gap sizes are re-randomised every example, so no fixed positional
  offset predicts the answer. This is what makes the task diagnostic of *addressing*
  rather than memorisation of layout.
- The generator is deliberately general. Literature-matched regimes (e.g. seq_len 256,
  n_pairs 4-64) belong in experiment configs, NOT in these defaults — Post 1 needs
  length/distance sweeps well outside any single canonical setting.
- Per-query metadata (`source_position`, `query_position`, `retrieval_distance`) is
  returned so evaluation can be *stratified* by retrieval distance. Do not build a
  separate "distance dataset"; generate ordinary MQAR and slice the results.

Token layout for a vocabulary of size V (ids are contiguous and disjoint):
    id 0                      reserved filler (used only if random_filler=False)
    [1, 1+n_key)              keys
    [1+n_key, 1+2*n_key)      values
where n_key = (V - 1) // 2.

Filler
------
By default (`random_filler=True`) gaps are filled with tokens drawn from the key and
value ranges, EXCLUDING the keys used in this example. Two properties follow:

  - the model cannot segment signal from noise by token identity alone, which a
    constant filler symbol would make trivial;
  - a queried key still occurs exactly twice (its presentation and its query), so the
    binding stays unambiguous.

Set `random_filler=False` for a constant-filler debugging variant.
"""

from dataclasses import dataclass, field

import numpy as np

IGNORE_INDEX = -100   # matches torch.nn.functional.cross_entropy default


@dataclass
class MQARSample:
    """One MQAR example.

    tokens:      (T,) int64 — the input sequence, answers included (teacher forcing)
    targets:     (T,) int64 — next-token targets, IGNORE_INDEX everywhere but answers
    loss_mask:   (T,) bool  — True exactly where targets != IGNORE_INDEX
    meta:        per-query bookkeeping for stratified evaluation
    """
    tokens: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    meta: dict = field(default_factory=dict)


def vocab_layout(vocab_size: int) -> dict:
    """Carve a vocabulary into filler / key / value ranges."""
    if vocab_size < 5:
        raise ValueError(f"vocab_size must be >= 5, got {vocab_size}")
    n_key = (vocab_size - 1) // 2
    return {
        "filler": 0,
        "key_lo": 1,
        "key_hi": 1 + n_key,              # exclusive
        "val_lo": 1 + n_key,
        "val_hi": 1 + 2 * n_key,          # exclusive
        "n_key": n_key,
    }


def min_seq_len(n_pairs: int, n_queries: int) -> int:
    """Shortest sequence that can hold the pairs and queries without overlap."""
    return 2 * n_pairs + 2 * n_queries


def generate_mqar(
    seq_len: int,
    n_pairs: int,
    vocab_size: int,
    n_queries: int = None,
    seed: int = None,
    rng: np.random.Generator = None,
    random_filler: bool = True,
) -> MQARSample:
    """Generate one MQAR example.

    Args:
        seq_len:    total sequence length
        n_pairs:    number of distinct key-value pairs presented
        vocab_size: total vocabulary (carved by `vocab_layout`)
        n_queries:  number of keys queried; defaults to n_pairs
        seed:       seed for a fresh generator (ignored if `rng` is given)
        rng:        an existing numpy Generator, for batch generation
        random_filler: fill gaps with non-key tokens (default) instead of a constant

    Returns:
        MQARSample
    """
    if n_queries is None:
        n_queries = n_pairs
    if rng is None:
        rng = np.random.default_rng(seed)

    lay = vocab_layout(vocab_size)
    if n_pairs > lay["n_key"]:
        raise ValueError(f"n_pairs={n_pairs} exceeds available keys ({lay['n_key']}); "
                         f"raise vocab_size")
    if n_queries > n_pairs:
        raise ValueError(f"n_queries={n_queries} > n_pairs={n_pairs}")
    need = min_seq_len(n_pairs, n_queries)
    if seq_len < need:
        raise ValueError(f"seq_len={seq_len} too short for n_pairs={n_pairs}, "
                         f"n_queries={n_queries} (need >= {need})")

    # Distinct keys; values sampled independently (repeats across keys are allowed,
    # which is standard — the binding, not the value identity, is what is tested).
    keys = rng.choice(np.arange(lay["key_lo"], lay["key_hi"]), size=n_pairs, replace=False)
    values = rng.integers(lay["val_lo"], lay["val_hi"], size=n_pairs)

    # Place 2-token blocks (pair or query) at random non-overlapping slots.
    n_blocks = n_pairs + n_queries
    # Choose block start positions so that no two blocks overlap: pick n_blocks
    # starts from the "compressed" axis then expand — standard trick for sampling
    # non-overlapping fixed-width intervals uniformly.
    free = seq_len - 2 * n_blocks
    starts = np.sort(rng.choice(np.arange(free + 1), size=n_blocks, replace=True))
    starts = starts + 2 * np.arange(n_blocks)

    # The first n_pairs blocks (in position order) present the pairs; the remaining
    # blocks query. A key can only be queried after it has been presented, so assign
    # presentation blocks to the earliest positions.
    pres_starts = starts[:n_pairs]
    query_starts = starts[n_pairs:]

    queried_idx = rng.choice(n_pairs, size=n_queries, replace=False)

    if random_filler:
        # Draw filler from key+value ids, minus this example's keys, so filler is
        # indistinguishable by identity yet cannot forge a second binding.
        pool = np.setdiff1d(np.arange(lay["key_lo"], lay["val_hi"]), keys)
        tokens = rng.choice(pool, size=seq_len).astype(np.int64)
    else:
        tokens = np.full(seq_len, lay["filler"], dtype=np.int64)
    targets = np.full(seq_len, IGNORE_INDEX, dtype=np.int64)
    loss_mask = np.zeros(seq_len, dtype=bool)

    for i, s in enumerate(pres_starts):
        tokens[s] = keys[i]
        tokens[s + 1] = values[i]

    q_meta = []
    for j, s in enumerate(query_starts):
        k = queried_idx[j]
        tokens[s] = keys[k]
        tokens[s + 1] = values[k]        # teacher forcing; loss is taken at s
        targets[s] = values[k]           # predict the value from the query key
        loss_mask[s] = True
        q_meta.append({
            "key": int(keys[k]),
            "value": int(values[k]),
            "source_position": int(pres_starts[k]),
            "query_position": int(s),
            "retrieval_distance": int(s - pres_starts[k]),
        })

    return MQARSample(
        tokens=tokens,
        targets=targets,
        loss_mask=loss_mask,
        meta={
            "queries": q_meta,
            "n_pairs": int(n_pairs),
            "n_queries": int(n_queries),
            "seq_len": int(seq_len),
            "vocab_size": int(vocab_size),
        },
    )


def make_batch(
    batch_size: int,
    seq_len: int,
    n_pairs: int,
    vocab_size: int,
    n_queries: int = None,
    seed: int = None,
    random_filler: bool = True,
):
    """Stack `batch_size` independent MQAR examples.

    Returns:
        tokens    (B, T) int64
        targets   (B, T) int64  (IGNORE_INDEX off-answer)
        loss_mask (B, T) bool
        metas     list[dict] of length B
    """
    rng = np.random.default_rng(seed)
    samples = [
        generate_mqar(seq_len, n_pairs, vocab_size, n_queries=n_queries, rng=rng,
                      random_filler=random_filler)
        for _ in range(batch_size)
    ]
    return (
        np.stack([s.tokens for s in samples]),
        np.stack([s.targets for s in samples]),
        np.stack([s.loss_mask for s in samples]),
        [s.meta for s in samples],
    )
