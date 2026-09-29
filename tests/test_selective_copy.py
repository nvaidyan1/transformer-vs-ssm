"""Property tests for the Selective Copying generator.

The defining property is randomised spacing: if a fixed set of taps could solve the
task, it would not distinguish a selective model from a time-invariant one, and the
implementation gate it is meant to provide would be worthless.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks.selective_copy import (DATA_LO, IGNORE_INDEX, MARKER_ID, NOISE_ID,
                                      generate_selective_copy, make_batch, min_seq_len)

T, ND, V = 48, 5, 16


def sample(seed, **kw):
    kw.setdefault("seq_len", T)
    kw.setdefault("n_data", ND)
    kw.setdefault("vocab_size", V)
    return generate_selective_copy(seed=seed, **kw)


# ── output region reproduces the data, in order ───────────────────────────────
def test_targets_are_the_data_in_order():
    for seed in range(50):
        s = sample(seed)
        m = s.meta["marker_position"]
        got = [int(s.targets[m + i]) for i in range(s.meta["n_data"])]
        assert got == s.meta["data"], "output must reproduce data in source order"


def test_teacher_forcing_alignment():
    """Predicting position p must mean 'the token at p+1'."""
    for seed in range(50):
        s = sample(seed)
        for p in np.flatnonzero(s.loss_mask):
            assert s.targets[p] == s.tokens[p + 1]


def test_loss_mask_covers_exactly_the_output_region():
    for seed in range(20):
        s = sample(seed)
        m, n = s.meta["marker_position"], s.meta["n_data"]
        assert set(np.flatnonzero(s.loss_mask)) == set(range(m, m + n))
        assert np.array_equal(s.loss_mask, s.targets != IGNORE_INDEX)


# ── the randomisation that makes the task diagnostic ──────────────────────────
def test_spacing_is_randomised():
    gaps = [g for seed in range(200) for g in sample(seed).meta["gaps"]]
    assert len(set(gaps)) > 5, "gaps look quantised"
    assert np.std(gaps) > 1.5, f"spacing nearly fixed: std={np.std(gaps):.2f}"


def test_no_fixed_source_offset_solves_the_task():
    """For each output slot i, the source position must vary across examples.

    If source position were a deterministic function of i, a fixed-offset (time
    invariant) model could solve the task and it would not gate selectivity.
    """
    by_slot = {i: set() for i in range(ND)}
    for seed in range(200):
        for i, p in enumerate(sample(seed).meta["source_positions"]):
            by_slot[i].add(p)
    for i, seen in by_slot.items():
        assert len(seen) > 10, f"output slot {i} reads a nearly fixed source position"


def test_data_positions_precede_the_marker():
    for seed in range(50):
        s = sample(seed)
        assert all(p < s.meta["marker_position"] for p in s.meta["source_positions"])
        assert s.tokens[s.meta["marker_position"]] == MARKER_ID


# ── vocabulary disjointness ───────────────────────────────────────────────────
def test_symbol_ranges_are_disjoint():
    for seed in range(50):
        s = sample(seed)
        assert all(d >= DATA_LO for d in s.meta["data"])
        inp = s.tokens[: s.meta["marker_position"]]
        noise_positions = set(range(s.meta["input_len"])) - set(s.meta["source_positions"])
        assert all(inp[p] == NOISE_ID for p in noise_positions)
        assert (s.tokens == MARKER_ID).sum() >= 1


# ── solvable, not degenerate ──────────────────────────────────────────────────
def test_oracle_scores_perfect():
    correct = total = 0
    for seed in range(100):
        s = sample(seed)
        m = s.meta["marker_position"]
        # oracle: take the non-noise tokens of the input region, in order
        recovered = [int(t) for t in s.tokens[:m] if t != NOISE_ID]
        for i in range(s.meta["n_data"]):
            correct += int(recovered[i] == s.targets[m + i])
            total += 1
    assert correct == total, f"oracle {correct}/{total} — task not solvable from the input"


def test_frequency_baseline_is_near_chance():
    from collections import Counter
    counts, answers = Counter(), []
    for seed in range(300):
        for d in sample(seed).meta["data"]:
            counts[d] += 1
            answers.append(d)
    guess = counts.most_common(1)[0][0]
    acc = float(np.mean([a == guess for a in answers]))
    n_symbols = V - DATA_LO
    assert acc < 4.0 / n_symbols, f"frequency baseline gets {acc:.3f} — task is degenerate"


# ── guards and batching ───────────────────────────────────────────────────────
def test_capacity_guards():
    with pytest.raises(ValueError):
        generate_selective_copy(seq_len=min_seq_len(ND) - 1, n_data=ND, vocab_size=V, seed=0)
    with pytest.raises(ValueError):
        generate_selective_copy(seq_len=T, n_data=ND, vocab_size=DATA_LO, seed=0)


def test_make_batch_shapes_seed_and_independence():
    tok, tgt, mask, metas = make_batch(8, T, ND, V, seed=1)
    assert tok.shape == tgt.shape == mask.shape == (8, T)
    assert len({tuple(m["source_positions"]) for m in metas}) > 1
    t2, _, _, _ = make_batch(8, T, ND, V, seed=1)
    assert np.array_equal(tok, t2), "same seed must reproduce the same batch"
    t3, _, _, _ = make_batch(8, T, ND, V, seed=2)
    assert not np.array_equal(tok, t3)


def test_train_val_seed_split_has_no_collisions():
    train = {sample(s).tokens.tobytes() for s in range(0, 400)}
    val = {sample(s).tokens.tobytes() for s in range(10_000, 10_400)}
    assert not (train & val)
