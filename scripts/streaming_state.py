"""
plan2.md 2.2: bytes carried between tokens during one-token-at-a-time generation, vs
context length, for the small matched configs (CLAIM_MAP D1). Measured by the steppers in
src/streaming.py (which reproduce each model's full forward; tests/test_streaming.py) and
computed analytically from shapes. Batch 1, float32, CPU.

The Transformer is built with max_seq_len 4096 (positional table only; the other weights
are as in transformer_small.yaml). The custom Mamba (mamba_small.yaml) stands in for
mamba_ssm: same state shapes (plan2 2.2 allows its per-step loop).

Usage:
    python scripts/streaming_state.py --out results/phase2/streaming_state.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.streaming import MambaStepper, TCNStepper, TransformerStepper, analytic_bytes
from src.train_task import write_record

CONTEXTS = (128, 256, 512, 1024, 2048, 4096)
SPECS = {
    "transformer": ("configs/recall/transformer_small.yaml", TransformerStepper, {"max_seq_len": 4096}),
    "tcn": ("configs/recall/tcn_small.yaml", TCNStepper, {}),
    "mamba": ("configs/recall/mamba_small.yaml", MambaStepper, {"scan": "sequential"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    args = ap.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    torch.manual_seed(0)
    rec = {"contexts": contexts, "batch": 1, "dtype": "float32", "archs": {}}
    for arch, (cfg_path, Stepper, over) in SPECS.items():
        cfg = yaml.safe_load(open(ROOT / cfg_path))
        model = build_model(arch, vocab_size=cfg["vocab_size"], **{**cfg["model"], **over})
        st = Stepper(model)
        toks = torch.randint(0, cfg["vocab_size"], (1, max(contexts)))
        measured, analytic, t0 = [], [], time.time()
        for t in range(max(contexts)):
            st.step(toks[:, t])
            if t + 1 in contexts:
                measured.append(st.state_bytes())
                analytic.append(analytic_bytes(arch, model, t + 1))
        extra = {"receptive_field": model.receptive_field} if arch == "tcn" else {}
        rec["archs"][arch] = {"config": cfg_path, "measured_bytes": measured,
                              "analytic_bytes": analytic, "seconds": round(time.time() - t0, 1),
                              **extra}
        print(arch, measured, f"{time.time() - t0:.0f}s", flush=True)
    write_record(rec, args.out)


if __name__ == "__main__":
    main()
