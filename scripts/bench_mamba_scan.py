"""
Benchmark the sequential scan (currently wired into src/models/mamba.py's
forward pass) against `_associative_scan` — a Hillis-Steele parallel prefix
scan that is fully implemented in the same file but never called anywhere.

This does NOT touch checkpoints or training. It is a pure correctness +
speed test of the recurrence

    h_t = a_t * h_{t-1} + b_t,   h_0 = 0

at shapes matching the real model (E=d_inner, N=d_state), so the result
tells us, before spending any more Kaggle GPU-hours, whether swapping the
scan implementation is a legitimate lever for Post 2's training-speed
problem.

Usage:
    python scripts/bench_mamba_scan.py
    python scripts/bench_mamba_scan.py --seq_lens 256 512 1024 --batch 4
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.mamba import _associative_scan


def sequential_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Naive python-loop recurrence — mirrors the loop in MambaSSM.forward."""
    B, T, E, N = a.shape
    state = torch.zeros(B, E, N, device=a.device, dtype=a.dtype)
    ys = []
    for t in range(T):
        state = a[:, t] * state + b[:, t]
        ys.append(state)
    return torch.stack(ys, dim=1)


def bench_one(seq_len: int, batch: int, d_inner: int, d_state: int, device: str):
    dev = torch.device(device)
    torch.manual_seed(0)
    # a in (0, 1) — matches A_bar = exp(negative), the real model's regime
    a = torch.rand(batch, seq_len, d_inner, d_state, device=dev) * 0.5 + 0.4
    b = torch.randn(batch, seq_len, d_inner, d_state, device=dev) * 0.1

    # Correctness
    with torch.no_grad():
        h_seq = sequential_reference(a, b)
        h_par = _associative_scan(a, b)
    max_diff = (h_seq - h_par).abs().max().item()

    # Speed — sequential
    n_reps = 3
    t0 = time.perf_counter()
    for _ in range(n_reps):
        with torch.no_grad():
            sequential_reference(a, b)
    t_seq = (time.perf_counter() - t0) / n_reps

    # Speed — parallel
    t0 = time.perf_counter()
    for _ in range(n_reps):
        with torch.no_grad():
            _associative_scan(a, b)
    t_par = (time.perf_counter() - t0) / n_reps

    return {
        "seq_len": seq_len,
        "max_diff": max_diff,
        "t_seq_ms": t_seq * 1000,
        "t_par_ms": t_par * 1000,
        "speedup": t_seq / t_par if t_par > 0 else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--batch", type=int, default=4, help="Small on purpose — parallel scan is memory-heavy.")
    parser.add_argument("--d_inner", type=int, default=512, help="Matches configs/mamba.yaml: d_model=256, expand=2.")
    parser.add_argument("--d_state", type=int, default=16, help="Matches configs/mamba.yaml.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}  batch={args.batch}  d_inner={args.d_inner}  d_state={args.d_state}")
    print(f"{'seq_len':>8}  {'max|diff|':>12}  {'seq (ms)':>10}  {'par (ms)':>10}  {'speedup':>8}")
    print("-" * 56)

    for seq_len in args.seq_lens:
        try:
            r = bench_one(seq_len, args.batch, args.d_inner, args.d_state, args.device)
            correct = "OK" if r["max_diff"] < 1e-4 else "MISMATCH"
            print(f"{r['seq_len']:>8}  {r['max_diff']:>10.2e}{'':<2}  "
                  f"{r['t_seq_ms']:>10.1f}  {r['t_par_ms']:>10.1f}  "
                  f"{r['speedup']:>7.2f}x  [{correct}]")
        except RuntimeError as e:
            print(f"{seq_len:>8}  FAILED: {e}")


if __name__ == "__main__":
    main()
