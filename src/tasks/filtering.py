"""
Scalar filtering tasks with exact baselines (plan2.md 5.1), for Post 3.

Latent random walk x_t = x_{t-1} + w_t, w_t ~ N(0, q), x_0 ~ N(0, P0). Observation
y_t = x_t + v_t, v_t ~ N(0, r_t). Target at every step: x_t. Inputs are continuous.

  stationary  fixed q and r. Input channel: y_t.
  switching   r_t switches between R_LO and R_HI in random blocks; a flag channel says
              which regime is active (1 = high noise). Inputs: y_t, flag_t.
  dropout     with probability P_DROP the observation is replaced by pure noise
              N(0, DROP_VAR), independent of x; a flag channel marks dropped steps.
              Inputs: y_t, flag_t.

Baselines, both given the true parameters:
  kalman()          exact scalar Kalman filter (gain K_t from the true q and r_t; K_t = 0
                    on a dropped step, whose observation carries no information about x).
  constant_gain()   x_hat_t = x_hat_{t-1} + k (y_t - x_hat_{t-1}) with one k for every
                    step; best_constant_gain() picks k by 1-D search on a training set.

In the stationary task the Kalman gain converges to a constant (steady-state Riccati), so
the best constant-gain filter is near-optimal there and a non-selective model suffices.
In the other two the optimal gain depends on the data (the flag), which is where
selectivity should matter. Parameters were chosen so the Kalman filter beats the best
constant gain by at least 20% on both (tests/test_filtering.py).
"""

import math

import numpy as np

Q = 0.01            # process noise variance, all tasks
P0 = 1.0            # prior variance of x_0 (prior mean 0)
R_STAT = 1.0        # stationary observation noise
R_LO, R_HI = 0.05, 5.0          # switching regimes
MEAN_BLOCK = 32                 # mean regime block length (geometric)
R_DROP, P_DROP, DROP_VAR = 0.1, 0.3, 4.0   # dropout task
TASKS = ("stationary", "switching", "dropout")


def generate(task: str, batch: int, seq_len: int, seed: int):
    """Returns inputs (B, T, C) float32, target x (B, T), r (B, T) true obs variance per
    step (np.inf on a dropped step), flag (B, T) in {0, 1}."""
    if task not in TASKS:
        raise ValueError(task)
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, math.sqrt(P0), size=(batch, 1))
    x = x0 + np.cumsum(rng.normal(0.0, math.sqrt(Q), size=(batch, seq_len)), axis=1)
    flag = np.zeros((batch, seq_len))
    if task == "stationary":
        r = np.full((batch, seq_len), R_STAT)
        y = x + rng.normal(size=x.shape) * math.sqrt(R_STAT)
    elif task == "switching":
        for b in range(batch):
            t, state = 0, rng.integers(2)
            while t < seq_len:
                n = rng.geometric(1.0 / MEAN_BLOCK)
                flag[b, t:t + n] = state
                t, state = t + n, 1 - state
        r = np.where(flag == 1, R_HI, R_LO)
        y = x + rng.normal(size=x.shape) * np.sqrt(r)
    else:
        flag = (rng.random((batch, seq_len)) < P_DROP).astype(float)
        noisy = x + rng.normal(size=x.shape) * math.sqrt(R_DROP)
        junk = rng.normal(size=x.shape) * math.sqrt(DROP_VAR)
        y = np.where(flag == 1, junk, noisy)
        r = np.where(flag == 1, np.inf, R_DROP)
    inputs = y[..., None] if task == "stationary" else np.stack([y, flag], axis=-1)
    return inputs.astype(np.float32), x.astype(np.float32), r, flag


def kalman(y: np.ndarray, r: np.ndarray, q: float = Q, p0: float = P0):
    """Exact scalar Kalman filter. y, r: (B, T). Returns estimates and gains, (B, T)."""
    B, T = y.shape
    est, gain = np.zeros((B, T)), np.zeros((B, T))
    m, P = np.zeros(B), np.full(B, p0)
    for t in range(T):
        Pp = P + q
        K = np.where(np.isinf(r[:, t]), 0.0, Pp / (Pp + np.where(np.isinf(r[:, t]), 1.0, r[:, t])))
        m = m + K * (y[:, t] - m)
        P = (1 - K) * Pp
        est[:, t], gain[:, t] = m, K
    return est, gain


def constant_gain(y: np.ndarray, k: float):
    B, T = y.shape
    est, m = np.zeros((B, T)), np.zeros(B)
    for t in range(T):
        m = m + k * (y[:, t] - m)
        est[:, t] = m
    return est


def best_constant_gain(y: np.ndarray, x: np.ndarray) -> float:
    """1-D search for the MSE-minimising constant gain: coarse grid, then refine."""
    def mse(k):
        return float(np.mean((constant_gain(y, k) - x) ** 2))
    grid = np.linspace(0.005, 1.0, 200)
    k = grid[int(np.argmin([mse(g) for g in grid]))]
    fine = np.linspace(max(1e-4, k - 0.01), min(1.0, k + 0.01), 81)
    return float(fine[int(np.argmin([mse(g) for g in fine]))])


def steady_state(q: float, r: float):
    """Closed-form steady state of the scalar random-walk Riccati recursion:
    prior variance P = (q + sqrt(q^2 + 4 q r)) / 2, gain K = P / (P + r), posterior
    variance (the steady-state MSE) = K r."""
    P = (q + math.sqrt(q * q + 4 * q * r)) / 2
    K = P / (P + r)
    return K, K * r
