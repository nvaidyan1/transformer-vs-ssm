"""Property tests for the MQAR generator.

These are invariants, not smoke tests: each one encodes a way the task could be
silently broken in a manner that would make downstream results uninterpretable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks.mqar import (IGNORE_INDEX, generate_mqar, make_batch, min_seq_len,
                            vocab_layout)

V = 65          # -> 32 keys, 32 values
T = 64
NP = 4


def sample(seed, **kw):
    kw.setdefault("seq_len", T)
    kw.setdefault("n_pairs", NP)
    kw.setdefault("vocab_size", V)
    return generate_mqar(seed=seed, **kw)


# ── 1. every queried key appeared previously in the same example ──────────────
def test_queried_key_presented_earlier():
    for seed in range(50):
        s = sample(seed)
        for q in s.meta["queries"]:
            assert q["source_position"] < q["query_position"]
            assert s.tokens[q["source_position"]] == q["key"]
            assert q["retrieval_distance"] > 0


# ── 2. the target is exactly the associated value ─────────────────────────────
def test_target_is_the_bound_value():
    for seed in range(50):
        s = sample(seed)
        for q in s.meta["queries"]:
            assert s.tokens[q["source_position"] + 1] == q["value"]
            assert s.targets[q["query_position"]] == q["value"]


# ── 3. key identities are unambiguous within an example ───────────────────────
def test_queried_key_occurs_exactly_twice():
    for seed in range(50):
        s = sample(seed)
        for q in s.meta["queries"]:
            assert int((s.tokens == q["key"]).sum()) == 2, (
                "a queried key must appear only at its presentation and its query; "
                "filler must never forge a second binding")


# ── 4. pair and query locations vary across examples ──────────────────────────
def test_layout_varies_across_examples():
    firsts = {sample(seed).meta["queries"][0]["query_position"] for seed in range(50)}
    assert len(firsts) > 10, "query positions look fixed across examples"


# ── 5. no fixed query->source offset leaks the solution ───────────────────────
def test_retrieval_distance_is_not_constant():
    d = [q["retrieval_distance"] for seed in range(100) for q in sample(seed).meta["queries"]]
    assert len(set(d)) > 15 and np.std(d) > 3.0, f"distance nearly constant: std={np.std(d):.2f}"


# ── 6. filler identity does not change the answer ─────────────────────────────
def test_targets_independent_of_filler_scheme():
    a = sample(7, random_filler=True)
    b = sample(7, random_filler=False)
    qa = [(q["key"], q["value"], q["source_position"], q["query_position"])
          for q in a.meta["queries"]]
    qb = [(q["key"], q["value"], q["source_position"], q["query_position"])
          for q in b.meta["queries"]]
    assert qa == qb, "filler scheme must not alter the bindings or their positions"


# ── 7. fresh associations, not a finite lookup table ──────────────────────────
def test_associations_are_freshly_generated():
    seen = set()
    for seed in range(200):
        for q in sample(seed).meta["queries"]:
            seen.add((q["key"], q["value"]))
    assert len(seen) > 200, f"only {len(seen)} distinct bindings — looks like a lookup table"


# ── 8. requested shape is achievable, and impossible shapes are rejected ──────
def test_capacity_guards():
    with pytest.raises(ValueError):
        generate_mqar(seq_len=min_seq_len(NP, NP) - 1, n_pairs=NP, vocab_size=V, seed=0)
    with pytest.raises(ValueError):
        generate_mqar(seq_len=T, n_pairs=vocab_layout(V)["n_key"] + 1, vocab_size=V, seed=0)
    with pytest.raises(ValueError):
        generate_mqar(seq_len=T, n_pairs=2, vocab_size=V, n_queries=3, seed=0)


def test_blocks_never_overlap():
    for seed in range(50):
        s = sample(seed)
        occupied = []
        for q in s.meta["queries"]:
            occupied += [q["source_position"], q["source_position"] + 1,
                         q["query_position"], q["query_position"] + 1]
        assert len(occupied) == len(set(occupied)), "pair/query blocks overlap"


# ── loss masking ──────────────────────────────────────────────────────────────
def test_loss_mask_covers_only_query_positions():
    for seed in range(20):
        s = sample(seed)
        qpos = {q["query_position"] for q in s.meta["queries"]}
        assert set(np.flatnonzero(s.loss_mask)) == qpos
        assert np.array_equal(s.loss_mask, s.targets != IGNORE_INDEX)
        assert s.loss_mask.sum() == s.meta["n_queries"]


# ── task is solvable, and not degenerate ──────────────────────────────────────
def test_oracle_scores_perfect():
    correct = total = 0
    for seed in range(100):
        s = sample(seed)
        for q in s.meta["queries"]:
            # oracle: scan the prefix for the key, read the next token
            pre = s.tokens[: q["query_position"]]
            hit = np.flatnonzero(pre == q["key"])
            correct += int(len(hit) == 1 and s.tokens[hit[0] + 1] == s.targets[q["query_position"]])
            total += 1
    assert correct == total, f"oracle {correct}/{total} — task not solvable from context alone"


def test_frequency_baseline_is_near_chance():
    """Always predicting the globally most common value must not beat chance."""
    from collections import Counter
    counts, answers = Counter(), []
    for seed in range(300):
        s = sample(seed)
        for q in s.meta["queries"]:
            counts[q["value"]] += 1
            answers.append(q["value"])
    guess = counts.most_common(1)[0][0]
    acc = np.mean([a == guess for a in answers])
    n_val = vocab_layout(V)["n_key"]
    assert acc < 4.0 / n_val, f"frequency baseline gets {acc:.3f} — task is degenerate"


# ── batching ──────────────────────────────────────────────────────────────────
def test_make_batch_shapes_and_independence():
    tok, tgt, mask, metas = make_batch(8, T, NP, V, seed=1)
    assert tok.shape == tgt.shape == mask.shape == (8, T)
    assert len(metas) == 8
    assert len({m["queries"][0]["query_position"] for m in metas}) > 1
    t2, _, _, _ = make_batch(8, T, NP, V, seed=1)
    assert np.array_equal(tok, t2), "same seed must reproduce the same batch"
    t3, _, _, _ = make_batch(8, T, NP, V, seed=2)
    assert not np.array_equal(tok, t3), "different seeds must differ"


def test_train_val_seed_split_has_no_collisions():
    train = {sample(s).tokens.tobytes() for s in range(0, 400)}
    val = {sample(s).tokens.tobytes() for s in range(10_000, 10_400)}
    assert not (train & val), "train/val sequence collision across the seed split"
