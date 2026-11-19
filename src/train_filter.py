"""
Train a small regression model on a filtering task (plan2.md 5.2); CPU is fine.

Arms: mamba (selective), mamba_ablated (input_independent=True), gru. The target is the
latent x_t at every step; loss and metric are MSE. A fixed validation set (seeds disjoint
from training) is evaluated every eval_every steps, alongside the exact Kalman filter and
the best constant-gain filter (gain fitted on a separate training draw) on the same set.

    python src/train_filter.py --task switching --arch mamba --n_layers 1 --seed 0 \
        --out results/phase5/switching_mamba_L1_s0.json --save_model results/phase5/m.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.regression import build_regression
from src.optim import build_param_groups
from src.tasks.filtering import TASKS, best_constant_gain, constant_gain, generate, kalman
from src.train_task import get_lr, set_seed, write_record

TRAIN_SEED_BASE = 0
VAL_SEED = 1_000_000_000
CONST_FIT_SEED = 2_000_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, required=True)
    ap.add_argument("--arch", choices=("mamba", "mamba_ablated", "gru"), required=True)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--schedule_steps", type=int, default=20000)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--val_batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scan", choices=("sequential", "parallel"), default="parallel")
    ap.add_argument("--max_minutes", type=float, default=None,
                    help="per-run time cap (plan2.md 5.2: 5 GPU-min); a run that hits it is "
                         "complete and marked stopped_by_time_cap")
    ap.add_argument("--save_model", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)      # several runs share the CPU
    dev = torch.device(args.device)
    set_seed(args.seed)
    in_dim = 1 if args.task == "stationary" else 2
    model = build_regression(args.arch, in_dim, args.d_model, args.n_layers, scan=args.scan).to(dev)
    kw = {"input_independent": True} if args.arch == "mamba_ablated" else {}
    opt = torch.optim.AdamW(build_param_groups(model, args.weight_decay), lr=args.lr,
                            betas=(0.9, 0.95))

    v_in, v_x, v_r, v_flag = generate(args.task, args.val_batch, args.seq_len, VAL_SEED)
    c_in, c_x, _, _ = generate(args.task, args.val_batch, args.seq_len, CONST_FIT_SEED)
    k_best = best_constant_gain(c_in[..., 0].astype(float), c_x)
    vy = v_in[..., 0].astype(float)
    kalman_mse = float(np.mean((kalman(vy, v_r)[0] - v_x) ** 2))
    const_mse = float(np.mean((constant_gain(vy, k_best) - v_x) ** 2))
    v_in_t, v_x_t = torch.from_numpy(v_in).to(dev), torch.from_numpy(v_x).to(dev)

    def val_mse():
        model.eval()
        with torch.no_grad():
            m = float(((model(v_in_t, **kw) - v_x_t) ** 2).mean())
        model.train()
        return m

    meta = {"task": args.task, "arch": args.arch, "n_layers": args.n_layers,
            "d_model": args.d_model, "seq_len": args.seq_len, "seed": args.seed,
            "lr": args.lr, "steps": args.steps, "schedule_steps": args.schedule_steps,
            "device": str(dev), "scan": args.scan if args.arch != "gru" else None,
            "n_params": sum(p.numel() for n, p in model.named_parameters()
                            if p.requires_grad and ".fixed_" not in n),
            "kalman_mse": kalman_mse, "const_gain_mse": const_mse, "const_gain_k": k_best}
    history, t0, capped = [], time.time(), False
    model.train()
    for step in range(args.steps):
        lr = get_lr(step, args.warmup_steps, args.schedule_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        u, x, _, _ = generate(args.task, args.batch_size, args.seq_len,
                              TRAIN_SEED_BASE + step + args.seed * 10_000_000)
        loss = ((model(torch.from_numpy(u).to(dev), **kw) - torch.from_numpy(x).to(dev)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            history.append({"step": step + 1, "train_mse": float(loss), "val_mse": val_mse()})
            write_record({**meta, "complete": False, "history": history,
                          "wall_clock_s": round(time.time() - t0, 1)}, args.out)
        if args.max_minutes and time.time() - t0 > 60 * args.max_minutes:
            capped = True
            history.append({"step": step + 1, "train_mse": float(loss), "val_mse": val_mse()})
            break
    final = val_mse()
    rec = {**meta, "complete": True, "stopped_by_time_cap": capped,
           "steps_run": history[-1]["step"] if history else 0,
           "history": history, "final_val_mse": final,
           "ratio_to_kalman": final / kalman_mse, "ratio_to_const_gain": final / const_mse,
           "wall_clock_s": round(time.time() - t0, 1)}
    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": args.arch, "task": args.task, "in_dim": in_dim,
                    "d_model": args.d_model, "n_layers": args.n_layers, "scan": args.scan,
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}},
                   args.save_model)
        rec["saved_model"] = args.save_model
    write_record(rec, args.out)
    print(f"{args.task} {args.arch} L{args.n_layers} s{args.seed}: val_mse {final:.4f} "
          f"(kalman {kalman_mse:.4f}, const {const_mse:.4f}) {rec['wall_clock_s']}s")


if __name__ == "__main__":
    main()
