#!/usr/bin/env bash
# Run the sequence-length sweep for all three architectures and save results.
# Output: checkpoints/sweep_results.json
#
# Usage:
#   bash scripts/sweep_length.sh
#   bash scripts/sweep_length.sh --batch_size 2   # override batch size

set -euo pipefail

BATCH_SIZE=${1:-4}
OUT="checkpoints/sweep_results.json"

mkdir -p checkpoints

python - <<EOF
import json, torch
from src.models import build_model
from src.evaluate import sweep_length

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}")

seq_lengths = [128, 256, 512, 1024, 2048, 4096]
batch_size  = $BATCH_SIZE
all_results = {}

for arch in ["transformer", "tcn", "mamba"]:
    print(f"sweeping {arch}...")
    model = build_model(arch).to(device)
    results = sweep_length(
        model,
        seq_lengths=seq_lengths,
        batch_size=batch_size,
        device=device,
        n_warmup=5,
        n_measure=20,
    )
    all_results[arch] = results
    for r in results:
        mem = f"{r['peak_memory_mb']:.1f} MB" if r['peak_memory_mb'] is not None else "N/A"
        time_str = f"{r['mean_time_ms']:.1f} ms" if r['mean_time_ms'] is not None else "OOM"
        print(f"  seq_len={r['seq_len']:>5d}  time={time_str:>10s}  mem={mem}")

with open("$OUT", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"Results saved to $OUT")
EOF
