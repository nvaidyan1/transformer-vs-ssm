"""
Shared training loop for transformer, TCN, and Mamba.

Usage:
    python src/train.py --config configs/transformer.yaml
    python src/train.py --config configs/transformer.yaml --max_steps 50 --log_every 10
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
import yaml

# Allow `python src/train.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_dataloader
from src.models import build_model, count_parameters


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: dict) -> dict:
    """Load YAML config and apply CLI overrides."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Flatten: top-level keys only (model: stays nested)
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train a byte-level language model.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    # Allow any top-level config key as a CLI override
    parser.add_argument("--max_steps",   type=int,   default=None)
    parser.add_argument("--log_every",   type=int,   default=None)
    parser.add_argument("--save_every",  type=int,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--seed",        type=int,   default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if k != "config" and v is not None}
    return args.config, overrides


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Device detection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── LR schedule ──────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, max_steps: int, lr: float) -> float:
    """Linear warmup then cosine decay to 10% of peak lr."""
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    """Full-pass evaluation. Returns loss and bpc."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        n = y.numel()
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    avg_loss = total_loss / total_tokens
    return {"loss": avg_loss, "bpc": avg_loss / math.log(2)}


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(ckpt_dir: Path, model: nn.Module, optimizer, step: int,
                    val_bpc: float, cfg: dict):
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "step": step,
        "val_bpc": val_bpc,
        "seed": cfg["seed"],
    }
    path = ckpt_dir / f"ckpt_{step:07d}.pt"
    torch.save(state, path)
    # Keep a stable "latest" symlink for easy resume detection
    latest = ckpt_dir / "latest.pt"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(path.name)
    return path


def find_latest_checkpoint(ckpt_dir: Path):
    """Return path to latest checkpoint, or None if none exist."""
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest.resolve()
    return None


def load_checkpoint(ckpt_path: Path, model: nn.Module, optimizer, device):
    """Load checkpoint in-place. Returns step and val_bpc."""
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    print(f"Resumed from {ckpt_path} (step {state['step']}, val_bpc {state['val_bpc']:.4f})")
    return state["step"], state["val_bpc"]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    config_path, overrides = parse_args()
    cfg = load_config(config_path, overrides)

    set_seed(cfg["seed"])
    device = get_device()

    _sep = "=" * 56
    print(_sep)
    print("  ENVIRONMENT")
    print(_sep)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / 1024 ** 3
        vram_free  = torch.cuda.mem_get_info(0)[0] / 1024 ** 3
        print(f"  GPU         : {props.name}")
        print(f"  VRAM        : {vram_free:.1f} GB free / {vram_total:.1f} GB total")
        print(f"  CUDA        : {torch.version.cuda}")
        print(f"  AMP         : enabled (float16, Tensor Cores active)")
        # Batch size guidance based on free VRAM
        if vram_free < 8:
            suggested_bs = 8
        elif vram_free < 12:
            suggested_bs = 16
        else:
            suggested_bs = 32
        print(f"  Suggested batch_size for this GPU: {suggested_bs} "
              f"(current: {cfg['batch_size']})")
    elif device.type == "mps":
        print(f"  GPU         : Apple MPS")
        print(f"  AMP         : disabled (MPS)")
    else:
        print(f"  Device      : CPU (no GPU detected)")
        print(f"  AMP         : disabled")
    print(_sep)
    print("  TRAINING CONFIG")
    print(_sep)
    print(f"  arch        : {cfg['arch']}")
    print(f"  seed        : {cfg['seed']}")
    print(f"  num_workers : {num_workers} (system CPUs: {multiprocessing.cpu_count()})")
    print(f"  pin_memory  : {pin_memory}")
    print(f"  batch_size  : {cfg['batch_size']}")
    print(f"  seq_len     : {cfg['seq_len']}")
    print(f"  max_steps   : {cfg['max_steps']:,}")
    print(f"  log_every   : {cfg['log_every']}")
    print(f"  grad_accum  : {cfg['grad_accum_steps']}")
    print(f"  checkpoint  : {cfg['checkpoint_dir']}")
    print(_sep)
    print("  NOTE: training auto-resumes if a checkpoint exists in")
    print("  the checkpoint dir — safe to restart interrupted runs.")
    print(_sep)

    # Build model
    model = build_model(cfg["arch"], **cfg["model"])
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"  params      : {n_params:,}")
    print(_sep)

    # Data loaders — cap workers to what the system actually has
    num_workers = min(cfg.get("num_workers", 4), multiprocessing.cpu_count())
    pin_memory  = cfg.get("pin_memory", True)
    train_loader = get_dataloader(
        "train",
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = get_dataloader(
        "val",
        seq_len=cfg["seq_len"],
        batch_size=cfg["batch_size"],
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    # Checkpoint directory
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    # AMP: enabled only on CUDA; MPS/CPU use full precision
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = device.type == "cuda"
    scaler = GradScaler(device=amp_device_type, enabled=use_amp)

    # Resume if a checkpoint exists
    step = 0
    last_val_bpc = float("inf")
    existing = find_latest_checkpoint(ckpt_dir)
    if existing:
        step, last_val_bpc = load_checkpoint(existing, model, optimizer, device)

    # ── Training loop ─────────────────────────────────────────────────────────
    max_steps      = cfg["max_steps"]
    warmup_steps   = cfg["warmup_steps"]
    grad_clip      = cfg["grad_clip"]
    grad_accum     = cfg["grad_accum_steps"]
    log_every      = cfg["log_every"]
    save_every     = cfg["save_every"]

    train_iter = iter(train_loader)
    accum_loss = 0.0

    model.train()
    t0 = time.time()

    while step < max_steps:
        # ── LR update ─────────────────────────────────────────────────────────
        lr_now = get_lr(step, warmup_steps, max_steps, cfg["lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad()

        # ── Gradient accumulation ─────────────────────────────────────────────
        for micro in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                set_seed(cfg["seed"] + step)   # deterministic reshuffle
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            with autocast(device_type=amp_device_type, enabled=use_amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                )
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # ── Gradient clip + step ──────────────────────────────────────────────
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        step += 1

        # ── Logging ───────────────────────────────────────────────────────────
        if step % log_every == 0:
            val_metrics = evaluate(model, val_loader, device)
            last_val_bpc = val_metrics["bpc"]
            elapsed = time.time() - t0
            train_bpc = (accum_loss / log_every) / math.log(2)
            print(
                f"step {step:>6d} | "
                f"lr {lr_now:.2e} | "
                f"train_loss {accum_loss:.4f} | "
                f"train_bpc {train_bpc:.4f} | "
                f"val_bpc {last_val_bpc:.4f} | "
                f"{elapsed:.1f}s"
            )
            accum_loss = 0.0
            t0 = time.time()

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step % save_every == 0:
            path = save_checkpoint(ckpt_dir, model, optimizer, step, last_val_bpc, cfg)
            print(f"  checkpoint → {path}")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    path = save_checkpoint(ckpt_dir, model, optimizer, step, last_val_bpc, cfg)
    print(f"Training complete. Final checkpoint → {path}")


if __name__ == "__main__":
    main()
