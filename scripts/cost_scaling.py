"""
plan2.md 2.1: peak GPU memory and wall-clock of one forward+backward pass at random init,
vs sequence length, for (a) the Transformer with naive attention (materialises T x T),
(b) the same Transformer with torch scaled_dot_product_attention, (c) the TCN and
(d) mamba_ssm. Small matched configs (CLAIM_MAP D1); the Transformer is built with
max_seq_len 4096 (positional table only). One fixed batch size for every model and length.

Per (model, length): 2 warmup passes, then 5 timed repeats with
torch.cuda.reset_peak_memory_stats() before each. Memory is reported both as peak
allocated and as peak minus the memory allocated just before the pass (model, inputs,
causal mask buffers); slopes are fitted on the second, the part that scales with length.
An out-of-memory pass is recorded as null.

Usage (CUDA; mamba_ssm needs mamba-ssm installed):
    python scripts/cost_scaling.py --out results/phase2/cost_scaling.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models import build_model, count_parameters
from src.train_task import write_record

LENGTHS = (128, 256, 512, 1024, 2048, 4096)
ARMS = {
    "transformer_naive": ("configs/recall/transformer_small.yaml", "transformer",
                          {"max_seq_len": 4096, "attn_impl": "naive"}),
    "transformer_sdpa": ("configs/recall/transformer_small.yaml", "transformer",
                         {"max_seq_len": 4096, "attn_impl": "sdpa"}),
    "tcn": ("configs/recall/tcn_small.yaml", "tcn", {}),
    "mamba_ssm": ("configs/recall/mamba_ssm_small.yaml", "mamba_ssm", {}),
}


def one_pass(model, x, y):
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    loss.backward()
    model.zero_grad(set_to_none=True)


def measure(model, vocab, L, batch, repeats, warmup):
    dev = torch.device("cuda")
    x = torch.randint(0, vocab, (batch, L), device=dev)
    y = torch.randint(0, vocab, (batch, L), device=dev)
    try:
        for _ in range(warmup):
            one_pass(model, x, y)
        times, peaks, deltas = [], [], []
        for _ in range(repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            t0 = time.perf_counter()
            one_pass(model, x, y)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            peak = torch.cuda.max_memory_allocated()
            peaks.append(peak)
            deltas.append(peak - base)
        return {"time_s": statistics.median(times), "time_s_all": times,
                "peak_bytes": max(peaks), "activation_bytes": max(deltas)}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"time_s": None, "peak_bytes": None, "activation_bytes": None, "oom": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    rec = {"lengths": list(LENGTHS), "batch": args.batch, "repeats": args.repeats,
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "arms": {}}
    for name in args.arms.split(","):
        cfg_path, arch, over = ARMS[name]
        cfg = yaml.safe_load(open(ROOT / cfg_path))
        torch.manual_seed(0)
        model = build_model(arch, vocab_size=cfg["vocab_size"],
                            **{**cfg["model"], **over}).cuda().train()
        rows = []
        for L in LENGTHS:
            r = measure(model, cfg["vocab_size"], L, args.batch, args.repeats, args.warmup)
            rows.append({"length": L, **r})
            print(name, L, r.get("time_s"), r.get("activation_bytes"), flush=True)
        rec["arms"][name] = {"config": cfg_path, "overrides": over,
                             "n_params": count_parameters(model), "rows": rows}
        del model
        torch.cuda.empty_cache()
        write_record(rec, args.out)
    write_record(rec, args.out)


if __name__ == "__main__":
    main()
