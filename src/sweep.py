"""
Sequence-length sweep: measure forward-pass time and peak memory for all
three architectures across increasing sequence lengths.

Two modes
---------
Quick (default) — runs in ~2–5 min, prints a diagnostic table, no file saved.
    python src/sweep.py

Full — runs in ~15–30 min, saves checkpoints/sweep_results.json for the
post notebooks.
    python src/sweep.py --full

The quick mode is intentionally lightweight so you can run it before committing
to a full training job and decide whether Mamba's sequential scan needs fixing.

Decision guide printed at the end of quick mode:
  - If Mamba time crosses below Transformer somewhere in the sweep range
    → O(n) story holds visually, no scan fix needed.
  - If Mamba is slower than Transformer at every seq_len
    → the efficiency claim is undermined; fix the parallel scan before training.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model
from src.evaluate import sweep_length


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_table(arch: str, results: list[dict]) -> None:
    print(f"\n  {arch.upper()}")
    print(f"  {'seq_len':>8}  {'time (ms)':>12}  {'±':>8}  {'mem (MB)':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*10}")
    for r in results:
        t   = f"{r['mean_time_ms']:.1f}"   if r["mean_time_ms"]    is not None else "OOM"
        std = f"{r['std_time_ms']:.1f}"    if r["std_time_ms"]     is not None else ""
        mem = f"{r['peak_memory_mb']:.1f}" if r["peak_memory_mb"]  is not None else "N/A"
        print(f"  {r['seq_len']:>8d}  {t:>12}  {std:>8}  {mem:>10}")


def decide(all_results: dict) -> None:
    """Print a plain-English recommendation based on timing crossover."""
    t_times = {r["seq_len"]: r["mean_time_ms"] for r in all_results["transformer"]
               if r["mean_time_ms"] is not None}
    m_times = {r["seq_len"]: r["mean_time_ms"] for r in all_results["mamba"]
               if r["mean_time_ms"] is not None}

    common = sorted(set(t_times) & set(m_times))
    crossover = None
    for sl in common:
        if m_times[sl] < t_times[sl]:
            crossover = sl
            break

    print("\n" + "=" * 56)
    print("  DECISION GUIDE")
    print("=" * 56)
    if crossover is not None:
        print(f"  Mamba is faster than Transformer at seq_len={crossover}.")
        print(f"  O(n) story holds visually.")
        print(f"  Recommendation: proceed to training as-is.")
    else:
        slowest_ratio = None
        for sl in common:
            ratio = m_times[sl] / t_times[sl]
            if slowest_ratio is None or ratio > slowest_ratio:
                slowest_ratio = ratio
                slowest_sl = sl
        print(f"  Mamba is slower than Transformer at every measured seq_len.")
        if slowest_ratio is not None:
            print(f"  Worst ratio: {slowest_ratio:.1f}x slower at seq_len={slowest_sl}.")
        print(f"  Recommendation: consider fixing the sequential scan in")
        print(f"  src/models/mamba.py before running the full training job.")
    print("=" * 56)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence-length sweep.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full sweep and save checkpoints/sweep_results.json.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for sweep forward passes (default: 4).",
    )
    parser.add_argument(
        "--out", type=str, default="checkpoints/sweep_results.json",
        help="Output path for full sweep JSON (default: checkpoints/sweep_results.json).",
    )
    args = parser.parse_args()

    device = get_device()

    if args.full:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]
        n_warmup, n_measure = 5, 20
        print(f"Full sweep  |  device: {device}  |  batch_size: {args.batch_size}")
    else:
        seq_lengths = [256, 512, 1024, 2048]
        n_warmup, n_measure = 3, 5
        print(f"Quick sweep  |  device: {device}  |  batch_size: {args.batch_size}")
        print(f"(run with --full to generate sweep_results.json for post notebooks)")

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  ({props.total_memory / 1024**3:.1f} GB)")

    all_results = {}
    for arch in ["transformer", "tcn", "mamba"]:
        print(f"\nSweeping {arch}...", flush=True)
        model = build_model(arch).to(device)
        results = sweep_length(
            model,
            seq_lengths=seq_lengths,
            batch_size=args.batch_size,
            device=device,
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
        all_results[arch] = results
        print_table(arch, results)

    if args.full:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved → {out_path}")
    else:
        decide(all_results)


if __name__ == "__main__":
    main()
