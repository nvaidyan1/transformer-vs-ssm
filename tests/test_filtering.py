"""plan2.md 5.1: filtering tasks and exact baselines (src/tasks/filtering.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks.filtering import (P_DROP, Q, R_HI, R_LO, R_STAT, TASKS, best_constant_gain,
                                 constant_gain, generate, kalman, steady_state)


def test_stationary_kalman_matches_closed_form_riccati():
    K, mse = steady_state(Q, R_STAT)
    inp, x, r, _ = generate("stationary", 2000, 512, seed=3)
    est, gains = kalman(inp[..., 0].astype(float), r)
    assert abs(gains[0, -1] / K - 1) < 0.02                       # gain converges
    emp = float(np.mean((est[:, 256:] - x[:, 256:]) ** 2))        # after the transient
    assert abs(emp / mse - 1) < 0.02


@pytest.mark.parametrize("task", ["switching", "dropout"])
def test_kalman_beats_best_constant_gain_by_20_percent(task):
    tr_inp, tr_x, _, _ = generate(task, 256, 256, seed=2)
    k = best_constant_gain(tr_inp[..., 0].astype(float), tr_x)
    inp, x, r, _ = generate(task, 256, 256, seed=1)
    y = inp[..., 0].astype(float)
    mk = np.mean((kalman(y, r)[0] - x) ** 2)
    mc = np.mean((constant_gain(y, k) - x) ** 2)
    assert mc >= 1.2 * mk, (task, mk, mc)


def test_shapes_flags_and_regimes():
    inp, x, r, flag = generate("stationary", 4, 64, seed=0)
    assert inp.shape == (4, 64, 1) and (flag == 0).all()
    inp, x, r, flag = generate("switching", 64, 256, seed=0)
    assert inp.shape == (64, 256, 2) and set(np.unique(r)) == {R_LO, R_HI}
    assert ((flag == 1) == (r == R_HI)).all() and (inp[..., 1] == flag).all()
    inp, x, r, flag = generate("dropout", 64, 256, seed=0)
    assert abs(flag.mean() - P_DROP) < 0.02 and np.isinf(r[flag == 1]).all()


def test_dropped_steps_get_zero_gain_and_generation_is_seeded():
    inp, x, r, flag = generate("dropout", 8, 64, seed=5)
    _, K = kalman(inp[..., 0].astype(float), r)
    assert (K[flag == 1] == 0).all() and (K[flag == 0] > 0).all()
    a, b = generate("switching", 2, 32, seed=7), generate("switching", 2, 32, seed=7)
    assert all(np.array_equal(u, v) for u, v in zip(a, b))
    assert set(TASKS) == {"stationary", "switching", "dropout"}
