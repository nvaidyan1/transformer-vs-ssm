"""
Train a small model on a synthetic diagnostic task (MQAR / Selective Copying).

Differs from src/train.py in three ways that matter:
  - data is generated on the fly, so there are no epochs — training is by steps;
  - the objective is masked to answer positions only (IGNORE_INDEX elsewhere), so
    filler cannot dominate the loss the way it did on enwik8;
  - the reported metric is exact-match accuracy at answer positions, not bpc.

Train and validation batches are drawn from DISJOINT seed ranges, and the validation
set is generated once and reused so curves are comparable across steps and runs.

Usage:
    python src/train_task.py --config configs/recall/mamba_small.yaml \
        --task selective_copy --task_args n_data=4,seq_len=128 --steps 2000

    # the selectivity ablation (Mamba only) — the Selective Copying gate control
    python src/train_task.py --config configs/recall/mamba_small.yaml \
        --task selective_copy --task_args n_data=4,seq_len=128 --ablate

    # the same model on the official mamba-ssm kernels (CUDA only), with the
    # original oracle notebook's harness settings
    python src/train_task.py --config configs/recall/mamba_small.yaml \
        --arch mamba_ssm --emb_init_std 1.0 --out_proj_depth_scale false \
        --decay_1d true --task selective_copy --task_args n_data=4,seq_len=64
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import build_model, count_parameters
from src.optim import build_param_groups, describe_param_groups
from src.tasks import IGNORE_INDEX, make_batcher

TRAIN_SEED_BASE = 0
VAL_SEED_BASE = 1_000_000_000   # disjoint from any training seed


def parse_task_args(s: str) -> dict:
    """Parse 'n_pairs=4,seq_len=128' into {'n_pairs': 4, 'seq_len': 128}."""
    out = {}
    if not s:
        return out
    for item in s.split(","):
        k, v = item.split("=", 1)
        out[k.strip()] = int(v) if v.strip().lstrip("-").isdigit() else v.strip()
    return out


def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))


MAMBA_ARCHS = ("mamba", "mamba_ssm")


def str2bool(s: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def count_ablation_only(model) -> int:
    """Parameters that exist only for the custom model's --ablate path."""
    return sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and ".fixed_" in n)


def write_record(record: dict, path: str):
    """Write atomically, so a run killed mid-write never leaves a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    tmp.replace(path)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_torch(batch, device):
    tok, tgt, mask, metas = batch
    return (torch.from_numpy(tok).to(device),
            torch.from_numpy(tgt).to(device),
            torch.from_numpy(mask).to(device),
            metas)


@torch.no_grad()
def evaluate(model, val_batches, device, model_kwargs) -> dict:
    """Exact-match accuracy and loss at answer positions."""
    model.eval()
    tot_loss = tot_correct = tot_n = 0
    for tok, tgt, mask, _ in val_batches:
        logits = model(tok, **model_kwargs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="sum")
        pred = logits.argmax(-1)
        tot_loss += loss.item()
        tot_correct += int((pred[mask] == tgt[mask]).sum().item())
        tot_n += int(mask.sum().item())
    model.train()
    return {"loss": tot_loss / max(1, tot_n),
            "acc": tot_correct / max(1, tot_n),
            "n": tot_n}


@torch.no_grad()
def answer_histograms(model, val_batches, model_kwargs) -> dict:
    """Counts of predicted and target tokens at answer positions over the validation
    set, as {token_id: count} (diagnostic only; plan2.md 1.1b)."""
    model.eval()
    pred, tgt_counts = {}, {}
    for tok, tgt, mask, _ in val_batches:
        p = model(tok, **model_kwargs).argmax(-1)[mask].tolist()
        t = tgt[mask].tolist()
        for k in p:
            pred[k] = pred.get(k, 0) + 1
        for k in t:
            tgt_counts[k] = tgt_counts.get(k, 0) + 1
    model.train()
    return {"pred": dict(sorted(pred.items())), "target": dict(sorted(tgt_counts.items()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", required=True, choices=("mqar", "selective_copy"))
    ap.add_argument("--task_args", default="", help="e.g. n_pairs=4,seq_len=128")
    ap.add_argument("--steps", type=int, default=None, help="override max_steps")
    ap.add_argument("--lr", type=float, default=None, help="override config lr")
    ap.add_argument("--ssm_lr_scale", type=float, default=None,
                    help="LR multiplier for dt_proj/x_proj/A_log (see src/optim.py). "
                         "Overrides config; default 1.0 if set nowhere.")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--arch", default=None,
                    help="override config arch, e.g. mamba_ssm with mamba_small.yaml")
    ap.add_argument("--emb_init_std", type=float, default=None,
                    help="Mamba archs: embedding init std (see apply_model_init).")
    ap.add_argument("--out_proj_depth_scale", type=str2bool, default=None,
                    help="Mamba archs: divide out_proj init by sqrt(n_layers).")
    ap.add_argument("--decay_1d", type=str2bool, default=None,
                    help="Weight-decay biases and norm gains too (see src/optim.py). "
                         "Overrides config; default false if set nowhere.")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ablate", action="store_true",
                    help="mamba / mamba_ssm: disable position-wise selectivity (gate control).")
    ap.add_argument("--early_stop_acc", type=float, default=None,
                    help="Stop once validation accuracy is strictly above this (e.g. "
                         "0.99) for --early_stop_evals consecutive evals.")
    ap.add_argument("--early_stop_evals", type=int, default=1,
                    help="Consecutive evals above --early_stop_acc required to stop.")
    ap.add_argument("--scan", choices=("sequential", "parallel"), default=None,
                    help="custom mamba only: scan implementation (numerically equivalent).")
    ap.add_argument("--grad_checkpoint", type=str2bool, default=None,
                    help="custom mamba only: activation checkpointing (memory/time only).")
    ap.add_argument("--dump_answer_hist", action="store_true",
                    help="Add histograms of predicted/target tokens at answer positions "
                         "(validation set, final model) to the result record.")
    ap.add_argument("--out", default=None, help="Write a JSON result record here.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg.setdefault("ssm_lr_scale", 1.0)
    cfg.setdefault("decay_1d", False)
    if args.arch is not None:
        cfg["arch"] = args.arch
    if args.decay_1d is not None:
        cfg["decay_1d"] = args.decay_1d
    for key in ("scan", "grad_checkpoint"):
        val = getattr(args, key)
        if val is not None:
            if cfg["arch"] != "mamba":
                raise SystemExit(f"--{key} only applies to the custom mamba arch")
            cfg["model"][key] = val
    for key in ("emb_init_std", "out_proj_depth_scale"):
        val = getattr(args, key)
        if val is not None:
            if cfg["arch"] not in MAMBA_ARCHS:
                raise SystemExit(f"--{key} only applies to {MAMBA_ARCHS}")
            cfg["model"][key] = val
    if args.steps is not None:
        cfg["max_steps"] = args.steps
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.ssm_lr_scale is not None:
        cfg["ssm_lr_scale"] = args.ssm_lr_scale

    task_kwargs = parse_task_args(args.task_args)
    task_kwargs.setdefault("vocab_size", cfg["vocab_size"])

    device = get_device(args.device)
    set_seed(cfg["seed"])

    arch = cfg["arch"]
    if args.ablate and arch not in MAMBA_ARCHS:
        raise SystemExit(f"--ablate is only meaningful for {MAMBA_ARCHS}")
    model_kwargs = {"input_independent": True} if args.ablate else {}

    model = build_model(arch, vocab_size=cfg["vocab_size"], **cfg["model"]).to(device)
    n_params = count_parameters(model)
    n_ablation_only = count_ablation_only(model)

    if not args.quiet:
        print(f"task   : {args.task}  {task_kwargs}")
        print(f"arch   : {arch}  ({n_params:,} params"
              + (f", {n_ablation_only:,} of them ablation-only" if n_ablation_only else "")
              + ")"
              + ("  [ABLATED: no position-wise selectivity]" if args.ablate else ""))
        print(f"device : {device}   lr={cfg['lr']}  steps={cfg['max_steps']}  "
              f"batch={cfg['batch_size']}  seed={cfg['seed']}")
        if arch in MAMBA_ARCHS:
            print(f"init   : emb_init_std={model.emb_init_std}  "
                  f"out_proj_depth_scale={model.out_proj_depth_scale}")
        print(describe_param_groups(model, ssm_lr_scale=cfg["ssm_lr_scale"],
                                    decay_1d=cfg["decay_1d"]))

    batcher = make_batcher(args.task, **task_kwargs)

    # Fixed validation set, from a seed range disjoint from training.
    val_batches = [to_torch(batcher(cfg["batch_size"], VAL_SEED_BASE + i), device)
                   for i in range(cfg["eval_batches"])]

    optimizer = torch.optim.AdamW(
        build_param_groups(model, cfg["weight_decay"], ssm_lr_scale=cfg["ssm_lr_scale"],
                           decay_1d=cfg["decay_1d"]),
        lr=cfg["lr"], betas=(0.9, 0.95))

    history, t0, stopped_early = [], time.time(), False
    streak = 0   # consecutive evals strictly above early_stop_acc

    def make_record(final, complete):
        return {
            "task": args.task, "task_args": task_kwargs, "arch": arch,
            "ablate": bool(args.ablate), "n_params": n_params,
            "n_params_ablation_only": n_ablation_only,
            "lr": cfg["lr"], "ssm_lr_scale": cfg["ssm_lr_scale"],
            "decay_1d": cfg["decay_1d"],
            "emb_init_std": getattr(model, "emb_init_std", None),
            "out_proj_depth_scale": getattr(model, "out_proj_depth_scale", None),
            "model_cfg": cfg["model"],
            "seed": cfg["seed"], "batch_size": cfg["batch_size"],
            "steps_run": history[-1]["step"] if history else 0,
            "max_steps": cfg["max_steps"], "stopped_early": stopped_early,
            "early_stop_acc": args.early_stop_acc,
            "early_stop_evals": args.early_stop_evals,
            "scan": model.blocks[0].ssm.scan if arch == "mamba" else None,
            "grad_checkpoint": getattr(model, "grad_checkpoint", None),
            # False in the checkpoints written at every eval; a run killed
            # mid-way (e.g. by a budget deadline) leaves its curve so far.
            "complete": complete,
            "final_val_acc": final["acc"] if final else None,
            "final_val_loss": final["loss"] if final else None,
            "chance_acc": 1.0 / max(1, (cfg["vocab_size"] - 1) // 2),
            "wall_clock_s": round(time.time() - t0, 1),
            "device": str(device), "history": history,
        }
    model.train()
    for step in range(cfg["max_steps"]):
        base_lr = get_lr(step, cfg["warmup_steps"], cfg["max_steps"], cfg["lr"])
        for g in optimizer.param_groups:
            # Every group carries its own lr_scale from build_param_groups (default
            # 1.0). Multiplying it in here, every step, is required: writing the
            # scheduled lr straight into g["lr"] would silently erase the multiplier
            # set at construction the moment the schedule first updates.
            g["lr"] = base_lr * g.get("lr_scale", 1.0)

        tok, tgt, _, _ = to_torch(
            batcher(cfg["batch_size"], TRAIN_SEED_BASE + step + cfg["seed"] * 10_000_000),
            device)
        logits = model(tok, **model_kwargs)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=IGNORE_INDEX)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

        if (step + 1) % cfg["eval_every"] == 0 or step == cfg["max_steps"] - 1:
            ev = evaluate(model, val_batches, device, model_kwargs)
            history.append({"step": step + 1, "train_loss": float(loss.item()),
                            "val_loss": ev["loss"], "val_acc": ev["acc"]})
            if not args.quiet:
                print(f"  step {step+1:>6d}  train_loss {loss.item():.4f}  "
                      f"val_loss {ev['loss']:.4f}  val_acc {ev['acc']:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if args.out:
                write_record(make_record(None, complete=False), args.out)
            if args.early_stop_acc is not None:
                streak = streak + 1 if ev["acc"] > args.early_stop_acc else 0
                if streak >= args.early_stop_evals:
                    stopped_early = True
                    if not args.quiet:
                        print(f"  early stop: val_acc > {args.early_stop_acc} for "
                              f"{streak} evals in a row")
                    break

    final = evaluate(model, val_batches, device, model_kwargs)
    record = make_record(final, complete=True)
    if args.dump_answer_hist:
        record["answer_hist"] = answer_histograms(model, val_batches, model_kwargs)
    if not args.quiet:
        print(f"\nfinal val_acc {final['acc']:.4f}  "
              f"(chance ~{record['chance_acc']:.4f})  in {record['wall_clock_s']}s")
    if args.out:
        write_record(record, args.out)
        if not args.quiet:
            print(f"saved -> {args.out}")
    return record


if __name__ == "__main__":
    main()
