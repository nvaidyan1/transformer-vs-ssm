"""
plan2.md 0.2 follow-up (amended): weight-copied custom vs official Mamba, 2000 steps.

The official model is built at the seed exactly as src/train_task.py builds it, its
weights are copied into a custom model, and both are trained in lockstep with
identical optimizers (same grouping, schedule, clip) on the identical batch at every
step. Both are evaluated on the same fixed validation set every eval_every steps.
The JSON record is rewritten at every eval (complete=false until the end).

Usage (CUDA + mamba-ssm):
    python scripts/phase0_weight_copy.py --emb_init_std 1.0 --decay_1d false \
        --seed 0 --out results/phase0_weight_copy/pair_emb1.0_d1d0_s0.json
"""

import argparse
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
from src.tasks import IGNORE_INDEX, chance_accuracy, make_batcher
from src.train_task import (TRAIN_SEED_BASE, VAL_SEED_BASE, evaluate, get_lr,
                            set_seed, str2bool, to_torch, write_record)


def train_pair(models: dict, cfg: dict, batcher, device, seed: int, steps: int,
               decay_1d: bool, on_eval=None) -> list:
    """Train every model in `models` (name -> nn.Module) in lockstep on identical
    batches with identically configured AdamW. Returns the eval history."""
    opts = {k: torch.optim.AdamW(build_param_groups(m, cfg["weight_decay"],
                                                    decay_1d=decay_1d),
                                 lr=cfg["lr"], betas=(0.9, 0.95))
            for k, m in models.items()}
    val = [to_torch(batcher(cfg["batch_size"], VAL_SEED_BASE + i), device)
           for i in range(cfg["eval_batches"])]
    history = []
    for m in models.values():
        m.train()
    for step in range(steps):
        base_lr = get_lr(step, cfg["warmup_steps"], steps, cfg["lr"])
        tok, tgt, _, _ = to_torch(
            batcher(cfg["batch_size"], TRAIN_SEED_BASE + step + seed * 10_000_000),
            device)
        losses = {}
        for k, m in models.items():
            for g in opts[k].param_groups:
                g["lr"] = base_lr * g.get("lr_scale", 1.0)
            logits = m(tok)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                ignore_index=IGNORE_INDEX)
            opts[k].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), cfg["grad_clip"])
            opts[k].step()
            losses[k] = float(loss.item())
        if (step + 1) % cfg["eval_every"] == 0 or step == steps - 1:
            row = {"step": step + 1}
            for k, m in models.items():
                ev = evaluate(m, val, device, {})
                row[k] = {"train_loss": losses[k], "val_loss": ev["loss"],
                          "val_acc": ev["acc"]}
            history.append(row)
            if on_eval:
                on_eval(history)
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/recall/mamba_small.yaml"))
    ap.add_argument("--emb_init_std", type=float, required=True)
    ap.add_argument("--decay_1d", type=str2bool, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--task_args", default="n_data=4,seq_len=64,vocab_size=64")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    task_kwargs = {k: int(v) for k, v in (kv.split("=") for kv in args.task_args.split(","))}
    device = torch.device(args.device)
    model_cfg = {**cfg["model"], "emb_init_std": args.emb_init_std,
                 "out_proj_depth_scale": True}

    from src.models.mamba_official import copy_official_to_custom
    set_seed(args.seed)                      # same order as src/train_task.py
    official = build_model("mamba_ssm", vocab_size=cfg["vocab_size"], **model_cfg).to(device)
    custom = build_model("mamba", vocab_size=cfg["vocab_size"], **model_cfg).to(device)
    copy_official_to_custom(official, custom)
    batcher = make_batcher("selective_copy", **task_kwargs)

    official.eval(); custom.eval()
    tok0 = to_torch(batcher(cfg["batch_size"], VAL_SEED_BASE), device)[0]
    with torch.no_grad():
        init_logit_diff = (official(tok0) - custom(tok0)).abs().max().item()
    print(f"max |logit diff| at step 0 after copy: {init_logit_diff:.3e}", flush=True)

    t0 = time.time()
    meta = {"emb_init_std": args.emb_init_std, "decay_1d": args.decay_1d,
            "out_proj_depth_scale": True, "seed": args.seed, "steps": args.steps,
            "task": "selective_copy", "task_args": task_kwargs, "model_cfg": model_cfg,
            "lr": cfg["lr"], "batch_size": cfg["batch_size"],
            "init_max_logit_diff": init_logit_diff, "device": str(device),
            "chance_acc": chance_accuracy("selective_copy", task_kwargs["vocab_size"])}

    def on_eval(history):
        h = history[-1]
        print(f"  step {h['step']:>5d}  official {h['official']['val_acc']:.4f}  "
              f"custom {h['custom']['val_acc']:.4f}  "
              f"diff {h['custom']['val_acc'] - h['official']['val_acc']:+.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        write_record({**meta, "complete": False, "history": history,
                      "wall_clock_s": round(time.time() - t0, 1)}, args.out)

    history = train_pair({"official": official, "custom": custom}, cfg, batcher,
                         device, args.seed, args.steps, args.decay_1d, on_eval)
    write_record({**meta, "complete": True, "history": history,
                  "wall_clock_s": round(time.time() - t0, 1)}, args.out)


if __name__ == "__main__":
    main()
