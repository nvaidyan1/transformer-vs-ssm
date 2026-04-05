"""
Evaluation utilities for byte-level language models.

Public API:
    evaluate(model, dataloader)                          -> {loss, bpc, ppl}
    evaluate_long_range(model, data, seq_len, min_distance) -> {long_range_bpc, n_positions}
    sweep_length(model, seq_lengths, batch_size, device) -> list[dict]
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn


# ── Standard evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, dataloader, device: torch.device = None) -> dict:
    """Evaluate model over a full dataloader.

    Args:
        model:      any model with interface logits = model(x)
        dataloader: yields (x, y) batches of LongTensors
        device:     if None, inferred from model parameters

    Returns:
        {'loss': float, 'bpc': float, 'ppl': float}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    model.train()
    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "bpc":  avg_loss / math.log(2),
        "ppl":  math.exp(avg_loss),
    }


# ── Long-range position detection ────────────────────────────────────────────

def _find_long_range_positions(data: np.ndarray, min_distance: int) -> np.ndarray:
    """Return a boolean mask over data marking positions that are long-range.

    A position i is "long-range" if the byte trigram data[i:i+3] was first seen
    at some position j where j <= i - min_distance.

    This is an approximation of "positions that require long-range context":
    a repeated trigram suggests the model benefits from remembering what came
    before. The approximation is conservative — it only flags positions where
    exact byte patterns recur, missing semantic dependencies that don't manifest
    as exact repeats. It is fast (one O(n) pass) and produces a reproducible,
    data-driven mask with no model involvement.

    Args:
        data:         uint8 numpy array (val split)
        min_distance: minimum number of bytes between first occurrence and
                      current position for the position to be flagged

    Returns:
        Boolean numpy array of shape (len(data),); True = long-range position.
    """
    n = len(data)
    mask = np.zeros(n, dtype=bool)
    first_seen = {}   # trigram bytes -> index of first occurrence

    for i in range(n - 2):
        trigram = (int(data[i]), int(data[i + 1]), int(data[i + 2]))
        if trigram not in first_seen:
            first_seen[trigram] = i
        elif i - first_seen[trigram] >= min_distance:
            mask[i] = True

    return mask


# ── Long-range evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_long_range(
    model: nn.Module,
    data: np.ndarray,
    seq_len: int,
    min_distance: int = 512,
    batch_size: int = 8,
    device: torch.device = None,
) -> dict:
    """Evaluate model loss restricted to long-range positions.

    Slices `data` into non-overlapping windows of length `seq_len` (same
    striding as ByteSequenceDataset), runs the model, and averages
    cross-entropy loss only over positions flagged as long-range.

    Args:
        model:        any model with interface logits = model(x)
        data:         uint8 numpy array (val split)
        seq_len:      sequence length for slicing (should match training)
        min_distance: minimum byte distance for a position to be long-range
        batch_size:   forward-pass batch size
        device:       if None, inferred from model parameters

    Returns:
        {'long_range_bpc': float, 'n_positions': int}
        n_positions is the number of tokens contributing to the metric.
    """
    if device is None:
        device = next(model.parameters()).device

    # Build long-range mask over the full data array
    lr_mask = _find_long_range_positions(data, min_distance)

    model.eval()
    total_loss = 0.0
    total_positions = 0

    # Non-overlapping windows: each window needs seq_len input + 1 target byte
    n_seqs = (len(data) - 1) // seq_len
    indices = list(range(n_seqs))

    for batch_start in range(0, len(indices), batch_size):
        batch_idx = indices[batch_start : batch_start + batch_size]

        # Build batch tensors
        xs, ys, masks = [], [], []
        for idx in batch_idx:
            start = idx * seq_len
            chunk = data[start : start + seq_len + 1].astype(np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            # mask[t] = True if position (start + t + 1) is long-range
            # (+1 because y[t] corresponds to data[start + t + 1])
            masks.append(lr_mask[start + 1 : start + seq_len + 1])

        x = torch.tensor(np.stack(xs), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(ys), dtype=torch.long, device=device)
        m = torch.tensor(np.stack(masks), dtype=torch.bool, device=device)

        if not m.any():
            continue

        logits = model(x)   # (B, T, vocab)
        # Per-token loss
        loss_per_token = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape(x.size(0), x.size(1))   # (B, T)

        total_loss += loss_per_token[m].sum().item()
        total_positions += m.sum().item()

    model.train()

    if total_positions == 0:
        return {"long_range_bpc": float("nan"), "n_positions": 0}

    avg_loss = total_loss / total_positions
    return {
        "long_range_bpc": avg_loss / math.log(2),
        "n_positions": total_positions,
    }


# ── Length sweep ──────────────────────────────────────────────────────────────

@torch.no_grad()
def sweep_length(
    model: nn.Module,
    seq_lengths: list,
    batch_size: int,
    device: str,
    n_warmup: int = 5,
    n_measure: int = 20,
) -> list:
    """Measure forward-pass time and peak memory at increasing sequence lengths.

    For each seq_len:
      - Run n_warmup forward passes (discarded).
      - Run n_measure forward passes and record wall-clock time.
      - Record peak GPU memory via torch.cuda.max_memory_allocated().
        On MPS or CPU, memory is recorded as None.
      - If an OOM error occurs, record None for all metrics at that seq_len.

    Args:
        model:       nn.Module already moved to device
        seq_lengths: list of sequence lengths to sweep
        batch_size:  batch size for each forward pass
        device:      'cuda', 'mps', or 'cpu'
        n_warmup:    number of warmup passes (not measured)
        n_measure:   number of measured passes

    Returns:
        List of dicts, one per seq_len:
        {'seq_len': int, 'mean_time_ms': float, 'std_time_ms': float,
         'peak_memory_mb': float or None}
    """
    dev = torch.device(device)
    is_cuda = dev.type == "cuda"
    results = []

    model.eval()

    for seq_len in seq_lengths:
        try:
            # Reset peak memory counter before warmup
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(dev)

            # Warmup
            for _ in range(n_warmup):
                x = torch.randint(0, 256, (batch_size, seq_len), device=dev)
                _ = model(x)
            if is_cuda:
                torch.cuda.synchronize(dev)

            # Reset again before measurement
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(dev)

            times = []
            for _ in range(n_measure):
                x = torch.randint(0, 256, (batch_size, seq_len), device=dev)
                if is_cuda:
                    torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                _ = model(x)
                if is_cuda:
                    torch.cuda.synchronize(dev)
                times.append((time.perf_counter() - t0) * 1000)  # ms

            peak_mb = (
                torch.cuda.max_memory_allocated(dev) / 1024 ** 2
                if is_cuda else None
            )

            results.append({
                "seq_len":       seq_len,
                "mean_time_ms":  float(np.mean(times)),
                "std_time_ms":   float(np.std(times)),
                "peak_memory_mb": peak_mb,
            })

        except torch.cuda.OutOfMemoryError:
            results.append({
                "seq_len":        seq_len,
                "mean_time_ms":   None,
                "std_time_ms":    None,
                "peak_memory_mb": None,
            })
            if is_cuda:
                torch.cuda.empty_cache()

    model.train()
    return results
