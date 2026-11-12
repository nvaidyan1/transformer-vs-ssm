"""
Stage A of plan2.md 1.1 (amended): time the custom Mamba's training step.

Model and optimizer exactly as src/train_task.py builds them from
configs/recall/mamba_small.yaml (selective_copy, n_data=4, seq_len=64, batch 32),
with the scan implementation and activation checkpointing switchable. Times
`--steps` full training steps (forward, masked loss, backward, clip, AdamW step)
after `--warmup` untimed steps, with CUDA synchronisation around the timed region.

    # one configuration
    python scripts/bench_custom_mamba.py --scan parallel --grad_checkpoint false \
        --out bench.json
    # several processes sharing the GPU: give each the same --start_at (epoch s)
    # so their timed regions overlap
    # parity of a configuration against the default (sequential, checkpointing on)
    python scripts/bench_custom_mamba.py --scan parallel --grad_checkpoint false \
        --parity --out parity.json
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.optim import build_param_groups
from src.tasks import IGNORE_INDEX, make_batcher
from src.train_task import TRAIN_SEED_BASE, set_seed, str2bool, to_torch, write_record

TASK = dict(n_data=4, seq_len=64, vocab_size=64)


def build(cfg, scan, grad_checkpoint, device):
    return build_model("mamba", vocab_size=cfg["vocab_size"],
                       **{**cfg["model"], "scan": scan,
                          "grad_checkpoint": grad_checkpoint}).to(device)


def set_mode(model, scan, grad_checkpoint):
    """Switch an existing model's execution mode without touching its weights."""
    for b in model.blocks:
        b.ssm.scan = scan
    model.grad_checkpoint = grad_checkpoint


def loss_of(model, tok, tgt):
    logits = model(tok)
    return logits, nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=IGNORE_INDEX)


def parity(cfg, scan, grad_checkpoint, device) -> dict:
    """Same weights, same batch: (scan, grad_checkpoint) vs (sequential, True)."""
    set_seed(0)
    ref = build(cfg, "sequential", True, device)
    test = copy.deepcopy(ref)
    set_mode(test, scan, grad_checkpoint)
    batcher = make_batcher("selective_copy", **TASK)
    tok, tgt, _, _ = to_torch(batcher(cfg["batch_size"], TRAIN_SEED_BASE), device)
    out = {}
    for name, m in (("ref", ref), ("test", test)):
        m.train()
        logits, loss = loss_of(m, tok, tgt)
        loss.backward()
        out[name] = (logits.detach(), loss.item(),
                     {n: p.grad.detach().clone() for n, p in m.named_parameters()
                      if p.grad is not None})
    (lr, sr, gr), (lt, st, gt) = out["ref"], out["test"]
    grad_abs = max((gr[n] - gt[n]).abs().max().item() for n in gr)
    grad_rel = max(((gr[n] - gt[n]).norm() / gr[n].norm().clamp_min(1e-30)).item()
                   for n in gr)
    return {"max_abs_logit_diff": (lr - lt).abs().max().item(),
            "loss_ref": sr, "loss_test": st, "max_abs_grad_diff": grad_abs,
            "max_rel_grad_diff": grad_rel, "n_grads": len(gr)}


def bench(cfg, scan, grad_checkpoint, steps, warmup, device, start_at=None) -> dict:
    set_seed(0)
    model = build(cfg, scan, grad_checkpoint, device)
    opt = torch.optim.AdamW(build_param_groups(model, cfg["weight_decay"]),
                            lr=cfg["lr"], betas=(0.9, 0.95))
    batcher = make_batcher("selective_copy", **TASK)
    batches = [to_torch(batcher(cfg["batch_size"], TRAIN_SEED_BASE + i), device)
               for i in range(warmup + steps)]
    model.train()

    def step(i):
        tok, tgt, _, _ = batches[i]
        _, loss = loss_of(model, tok, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

    for i in range(warmup):
        step(i)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    if start_at is not None:
        time.sleep(max(0.0, start_at - time.time()))
    t0 = time.time()
    for i in range(warmup, warmup + steps):
        step(i)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return {"steps": steps, "seconds": dt, "steps_per_s": steps / dt,
            "ms_per_step": 1000 * dt / steps, "t_start": t0,
            "peak_mem_mib": (torch.cuda.max_memory_allocated() / 2**20
                             if device.type == "cuda" else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/recall/mamba_small.yaml"))
    ap.add_argument("--scan", choices=("sequential", "parallel"), required=True)
    ap.add_argument("--grad_checkpoint", type=str2bool, required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--start_at", type=float, default=None)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device)
    rec = {"scan": args.scan, "grad_checkpoint": args.grad_checkpoint, "tag": args.tag,
           "device": str(device), "batch_size": cfg["batch_size"], "task": TASK,
           "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    if args.parity:
        rec["parity"] = parity(cfg, args.scan, args.grad_checkpoint, device)
    else:
        rec.update(bench(cfg, args.scan, args.grad_checkpoint, args.steps, args.warmup,
                         device, args.start_at))
    write_record(rec, args.out)
    print(json.dumps({k: v for k, v in rec.items() if k != "task"}))


if __name__ == "__main__":
    main()
