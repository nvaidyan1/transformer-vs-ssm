"""
plan2.md 5.3: does a trained model's selection behave like a Kalman gain?

effective_gain(model, u): d estimate_t / d observation_t for every step, by autograd (one
backward pass per step; the models are causal, so this is the Jacobian diagonal).
delta_by_regime(model, u, flag): mean softplus Delta per layer and channel, split by the
noise-regime flag (selective RegMamba only).
"""

import numpy as np
import torch


def effective_gain(model, u: torch.Tensor, **kw) -> np.ndarray:
    """u: (B, T, C), observation in channel 0. Returns (B, T)."""
    model.eval()
    u = u.clone().requires_grad_(True)
    # cuDNN refuses RNN backward in eval mode; the models have no dropout, so turning
    # cuDNN off here changes nothing but the kernel used (found on Kaggle, 2026-11-19).
    with torch.backends.cudnn.flags(enabled=False):
        out = model(u, **kw)                               # (B, T)
    B, T = out.shape
    g = np.zeros((B, T))
    for t in range(T):
        (grad,) = torch.autograd.grad(out[:, t].sum(), u, retain_graph=t < T - 1)
        g[:, t] = grad[:, t, 0].detach().cpu().numpy()
    return g


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


@torch.no_grad()
def delta_by_regime(model, u: torch.Tensor, flag: np.ndarray) -> list:
    """Per layer: mean Delta per channel in the low (flag 0) and high (flag 1) regimes."""
    model.eval()
    _, deltas = model(u, return_delta=True)
    f = torch.as_tensor(flag, dtype=torch.bool, device=u.device)
    out = []
    for d in deltas:                                       # (B, T, E)
        lo = d[~f].mean(0).cpu().numpy()                   # (E,)
        hi = d[f].mean(0).cpu().numpy()
        out.append({"low_per_channel": lo.tolist(), "high_per_channel": hi.tolist(),
                    "low_mean": float(lo.mean()), "high_mean": float(hi.mean()),
                    "low_channel_std": float(lo.std()), "high_channel_std": float(hi.std()),
                    "frac_channels_higher_in_low_noise": float((lo > hi).mean())})
    return out
