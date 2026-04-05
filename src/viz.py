"""
Figure generation for the Latent Dynamics blog series.

Every public function:
  - Takes a checkpoint path (or pre-loaded data) and an output path
  - Loads model / data as needed
  - Saves the figure to output_path at 150 DPI as PNG
  - Returns output_path

Style conventions (per blog spec):
  - Seaborn whitegrid
  - No figure titles (captions live in the post markdown)
  - transformer = blue (#4477AA), TCN = orange (#EE7733), Mamba = green (#228833)
  - Font legible at ~700px blog column width
  - 150 DPI PNG
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless/server use
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from src.models import build_model

# ── Style constants ───────────────────────────────────────────────────────────

COLORS = {
    "transformer": "#4477AA",
    "tcn":         "#EE7733",
    "mamba":       "#228833",
}
DPI = 150
sns.set_theme(style="whitegrid", font_scale=1.1)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_model(ckpt_path: str, device: torch.device):
    """Load a model from a checkpoint. Returns (model, cfg)."""
    state = torch.load(ckpt_path, map_location=device)
    cfg = state["config"]
    model = build_model(cfg["arch"], **cfg["model"])
    model.load_state_dict(state["model_state"])
    model.eval()
    model.to(device)
    return model, cfg


def _bytes_to_labels(data: bytes | list) -> list[str]:
    """Convert a sequence of byte values to display strings.

    Printable ASCII is shown as-is; non-printable bytes shown as '·'.
    """
    labels = []
    for b in data:
        ch = chr(b) if isinstance(b, int) else chr(ord(b))
        labels.append(ch if ch.isprintable() and ch != " " else "·")
    return labels


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig, output_path: str) -> str:
    path = _ensure_dir(output_path)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ── Post A — Figure 1: Attention heatmap ─────────────────────────────────────

def fig_attention_heatmap(
    ckpt_path: str,
    sequences: list[str],
    output_path: str,
    device: str = "cpu",
) -> str:
    """Attention weight heatmap for the transformer.

    Args:
        ckpt_path:   path to transformer checkpoint
        sequences:   list of short strings to visualise
        output_path: where to save the PNG
        device:      torch device string

    Returns:
        output_path
    """
    dev = torch.device(device)
    model, _ = _load_model(ckpt_path, dev)

    n_seq = len(sequences)
    fig, axes = plt.subplots(1, n_seq, figsize=(5 * n_seq, 5))
    if n_seq == 1:
        axes = [axes]

    with torch.no_grad():
        for ax, seq in zip(axes, sequences):
            x = torch.tensor(
                [[b for b in seq.encode("utf-8")]],
                dtype=torch.long,
                device=dev,
            )
            _, all_attn = model(x, return_attention=True)

            # Average attention across layers and heads: (T, T)
            attn = torch.stack(all_attn, dim=0).mean(dim=0)  # (1, H, T, T)
            attn = attn.squeeze(0).mean(0).cpu().numpy()      # (T, T)

            labels = _bytes_to_labels(seq.encode("utf-8"))
            sns.heatmap(
                attn,
                ax=ax,
                xticklabels=labels,
                yticklabels=labels,
                cmap="Blues",
                vmin=0,
                cbar=True,
                square=True,
            )
            ax.set_xlabel("Key position")
            ax.set_ylabel("Query position")
            ax.tick_params(axis="both", labelsize=8)

    fig.tight_layout()
    return _save(fig, output_path)


# ── Post A — Figure 2: Transformer length sweep ───────────────────────────────

def fig_length_sweep_transformer(
    sweep_results: list[dict],
    output_path: str,
) -> str:
    """Time and memory vs seq_len for the transformer on log-log axes.

    Overlays a theoretical O(n²) line scaled to match at the smallest
    non-None data point.

    Args:
        sweep_results: list of dicts from sweep_length() for transformer
        output_path:   where to save the PNG

    Returns:
        output_path
    """
    valid = [r for r in sweep_results if r["mean_time_ms"] is not None]
    seq_lens = np.array([r["seq_len"] for r in valid])
    times    = np.array([r["mean_time_ms"] for r in valid])
    stds     = np.array([r["std_time_ms"] for r in valid])

    has_memory = any(r["peak_memory_mb"] is not None for r in valid)
    mems = (
        np.array([r["peak_memory_mb"] for r in valid])
        if has_memory else None
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    color = COLORS["transformer"]

    # ── Time panel ────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.errorbar(seq_lens, times, yerr=stds, fmt="o-", color=color,
                linewidth=2, capsize=4, label="Transformer")
    # O(n²) overlay scaled at first data point
    scale = times[0] / seq_lens[0] ** 2
    n2 = scale * seq_lens ** 2
    ax.plot(seq_lens, n2, "k--", linewidth=1, label=r"$O(n^2)$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Time per forward pass (ms)")
    ax.legend(frameon=False)

    # ── Memory panel ─────────────────────────────────────────────────────────
    ax = axes[1]
    if mems is not None:
        ax.plot(seq_lens, mems, "o-", color=color, linewidth=2,
                label="Transformer")
        scale_m = mems[0] / seq_lens[0] ** 2
        ax.plot(seq_lens, scale_m * seq_lens ** 2, "k--", linewidth=1,
                label=r"$O(n^2)$")
        ax.set_yscale("log")
        ax.set_ylabel("Peak memory (MB)")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "Memory not available\n(non-CUDA device)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="grey")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length")

    fig.tight_layout()
    return _save(fig, output_path)


# ── Post A — Figure 3: TCN vs Transformer ─────────────────────────────────────

def fig_tcn_vs_transformer(
    transformer_sweep: list[dict],
    tcn_sweep: list[dict],
    output_path: str,
    eval_results: dict | None = None,
) -> str:
    """Four-panel comparison: memory, time, overall bpc, long-range bpc.

    Args:
        transformer_sweep: sweep_length() output for transformer
        tcn_sweep:         sweep_length() output for TCN
        output_path:       where to save the PNG
        eval_results:      optional dict with keys 'transformer' and 'tcn',
                           each a dict with 'bpc', 'long_range_bpc', 'n_positions'
                           (from evaluate() + evaluate_long_range())

    Returns:
        output_path
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for arch, sweep, ax_mem, ax_time in [
        ("transformer", transformer_sweep, axes[0][0], axes[0][1]),
        ("tcn",         tcn_sweep,         axes[0][0], axes[0][1]),
    ]:
        valid = [r for r in sweep if r["mean_time_ms"] is not None]
        if not valid:
            continue
        sl = np.array([r["seq_len"] for r in valid])
        t  = np.array([r["mean_time_ms"] for r in valid])
        c  = COLORS[arch]

        ax_time.plot(sl, t, "o-", color=c, linewidth=2, label=arch.upper())

        if valid[0]["peak_memory_mb"] is not None:
            m = np.array([r["peak_memory_mb"] for r in valid])
            ax_mem.plot(sl, m, "o-", color=c, linewidth=2, label=arch.upper())

    for ax in [axes[0][0], axes[0][1]]:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Sequence length")
        ax.legend(frameon=False)
    axes[0][0].set_ylabel("Peak memory (MB)")
    axes[0][1].set_ylabel("Time per forward pass (ms)")

    # ── BPC panels ────────────────────────────────────────────────────────────
    ax_bpc      = axes[1][0]
    ax_lr_bpc   = axes[1][1]

    if eval_results:
        archs = ["transformer", "tcn"]
        bpcs    = [eval_results[a].get("bpc", 0)            for a in archs]
        lr_bpcs = [eval_results[a].get("long_range_bpc", 0) for a in archs]
        colors  = [COLORS[a] for a in archs]

        ax_bpc.bar(archs, bpcs, color=colors, width=0.5)
        ax_bpc.set_ylabel("bits per character")

        ax_lr_bpc.bar(archs, lr_bpcs, color=colors, width=0.5)
        ax_lr_bpc.set_ylabel("long-range bpc")
    else:
        for ax, label in [(ax_bpc, "overall bpc"), (ax_lr_bpc, "long-range bpc")]:
            ax.text(0.5, 0.5, f"{label}\n(run after training)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="grey")

    fig.tight_layout()
    return _save(fig, output_path)


# ── Post B — Figure 1: All three sweep ───────────────────────────────────────

def fig_all_three_sweep(
    transformer_sweep: list[dict],
    tcn_sweep: list[dict],
    mamba_sweep: list[dict],
    output_path: str,
) -> str:
    """Memory and time curves for all three architectures.

    Overlays theoretical O(n²) and O(n) lines.

    Args:
        transformer_sweep: sweep_length() output for transformer
        tcn_sweep:         sweep_length() output for TCN
        mamba_sweep:       sweep_length() output for Mamba
        output_path:       where to save the PNG

    Returns:
        output_path
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sweeps = [
        ("transformer", transformer_sweep),
        ("tcn",         tcn_sweep),
        ("mamba",       mamba_sweep),
    ]

    ref_sl = ref_t = ref_m = None   # for scaling O(n²) / O(n) overlays

    for arch, sweep in sweeps:
        valid = [r for r in sweep if r["mean_time_ms"] is not None]
        if not valid:
            continue
        sl = np.array([r["seq_len"] for r in valid])
        t  = np.array([r["mean_time_ms"] for r in valid])
        c  = COLORS[arch]

        axes[1].plot(sl, t, "o-", color=c, linewidth=2, label=arch.capitalize())

        if valid[0]["peak_memory_mb"] is not None:
            m = np.array([r["peak_memory_mb"] for r in valid])
            axes[0].plot(sl, m, "o-", color=c, linewidth=2,
                         label=arch.capitalize())
            if ref_sl is None:
                ref_sl, ref_t, ref_m = sl, t, m

    # Overlay O(n²) and O(n) on time panel, scaled to transformer at seq_len=256
    if ref_sl is not None:
        idx = np.where(ref_sl == 256)[0]
        if len(idx):
            i = idx[0]
            n2_scale = ref_t[i] / 256 ** 2
            n1_scale = ref_t[i] / 256
            sl_line = np.array([128, 256, 512, 1024, 2048, 4096])
            axes[1].plot(sl_line, n2_scale * sl_line ** 2, "k--",
                         linewidth=1, label=r"$O(n^2)$")
            axes[1].plot(sl_line, n1_scale * sl_line, "k:",
                         linewidth=1, label=r"$O(n)$")

        if ref_m is not None:
            i = np.where(ref_sl == 256)[0]
            if len(i):
                m2_scale = ref_m[i[0]] / 256 ** 2
                m1_scale = ref_m[i[0]] / 256
                sl_line = np.array([128, 256, 512, 1024, 2048, 4096])
                axes[0].plot(sl_line, m2_scale * sl_line ** 2, "k--",
                             linewidth=1, label=r"$O(n^2)$")
                axes[0].plot(sl_line, m1_scale * sl_line, "k:",
                             linewidth=1, label=r"$O(n)$")

    for ax, ylabel in zip(axes, ["Peak memory (MB)", "Time per forward pass (ms)"]):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Sequence length")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    return _save(fig, output_path)


# ── Post B — Figure 2: Perplexity comparison ─────────────────────────────────

def fig_perplexity_comparison(
    results_dict: dict,
    output_path: str,
) -> str:
    """Side-by-side bar chart: overall bpc and long-range bpc for all three.

    Args:
        results_dict: {arch: {'bpc': float, 'long_range_bpc': float, ...}}
                      Keys should include 'transformer', 'tcn', 'mamba'.
        output_path:  where to save the PNG

    Returns:
        output_path
    """
    archs   = [a for a in ["transformer", "tcn", "mamba"] if a in results_dict]
    bpcs    = [results_dict[a]["bpc"]            for a in archs]
    lr_bpcs = [results_dict[a]["long_range_bpc"] for a in archs]
    colors  = [COLORS[a] for a in archs]

    x = np.arange(len(archs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width / 2, bpcs,    width, color=colors,
                   alpha=0.9, label="Overall bpc")
    bars2 = ax.bar(x + width / 2, lr_bpcs, width, color=colors,
                   alpha=0.5, label="Long-range bpc", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in archs])
    ax.set_ylabel("bits per character")
    ax.legend(frameon=False)

    fig.tight_layout()
    return _save(fig, output_path)


# ── Post B — Figure 3: Delta-t visualisation ─────────────────────────────────

def fig_delta_t(
    ckpt_path: str,
    sequence: str,
    output_path: str,
    device: str = "cpu",
    annotate: list[tuple[int, str]] | None = None,
) -> str:
    """Plot Mamba's delta_t (selectivity gate) across a sequence.

    delta_t controls how strongly the SSM state is updated at each position.
    Higher delta_t means the model gates more information from that token into
    its hidden state.  Caption: "Higher delta_t = state updates more strongly
    from this token."

    Args:
        ckpt_path:  path to Mamba checkpoint
        sequence:   string to run through the model
        output_path: where to save the PNG
        device:     torch device string
        annotate:   optional list of (position_index, label) tuples for
                    vertical annotation lines (e.g. key nouns, fillers)

    Returns:
        output_path
    """
    dev = torch.device(device)
    model, _ = _load_model(ckpt_path, dev)

    x = torch.tensor(
        [[b for b in sequence.encode("utf-8")]],
        dtype=torch.long,
        device=dev,
    )

    with torch.no_grad():
        _, all_deltas = model(x, return_delta=True)

    # all_deltas: list of (1, T, d_inner) per layer
    # Average over layers and d_inner → (T,)
    delta = torch.stack(all_deltas, dim=0).squeeze(1)  # (n_layers, T, d_inner)
    delta_mean = delta.mean(dim=(0, 2)).cpu().numpy()   # (T,)

    labels = _bytes_to_labels(sequence.encode("utf-8"))
    T = len(labels)
    positions = np.arange(T)

    fig, ax = plt.subplots(figsize=(max(8, T * 0.3), 3.5))
    ax.plot(positions, delta_mean, color=COLORS["mamba"], linewidth=1.5)
    ax.fill_between(positions, delta_mean, alpha=0.2, color=COLORS["mamba"])

    if annotate:
        for pos, label in annotate:
            ax.axvline(pos, color="black", linestyle="--", linewidth=1,
                       alpha=0.7)
            ax.text(pos + 0.2, ax.get_ylim()[1] * 0.95, label,
                    fontsize=8, va="top")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel(r"Mean $\Delta_t$")
    ax.set_xlabel("Token position")

    # Caption note embedded as figure text
    fig.text(
        0.01, -0.02,
        "Higher \u0394t = state updates more strongly from this token",
        fontsize=8, color="grey",
    )

    fig.tight_layout()
    return _save(fig, output_path)


# ── Post B — Figure 4: Training / inference schematic ────────────────────────

def fig_training_inference_schematic(output_path: str) -> str:
    """Drawn schematic: parallel scan (training) vs sequential recurrence (inference).

    No model or checkpoint required.

    Returns:
        output_path
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ── Left panel: parallel associative scan (training) ─────────────────────
    ax = axes[0]
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    node_color = COLORS["mamba"]
    token_color = "#DDDDDD"

    # Token layer (bottom)
    token_xs = [1, 3, 5, 7]
    tokens = ["x1", "x2", "x3", "x4"]
    for x, label in zip(token_xs, tokens):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.5, 0.2), 1.0, 0.8,
            boxstyle="round,pad=0.1", linewidth=1.2,
            edgecolor="black", facecolor=token_color,
        ))
        ax.text(x, 0.6, label, ha="center", va="center", fontsize=11)

    # Level 1: pairwise reductions
    level1_xs = [2, 6]
    level1_labels = ["h12", "h34"]
    for x, label, (lx, rx) in zip(
        level1_xs, level1_labels, [(1, 3), (5, 7)]
    ):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.5, 1.7), 1.0, 0.8,
            boxstyle="round,pad=0.1", linewidth=1.2,
            edgecolor="black", facecolor=node_color, alpha=0.7,
        ))
        ax.text(x, 2.1, label, ha="center", va="center", fontsize=10,
                color="white", fontweight="bold")
        for tx in [lx, rx]:
            ax.annotate(
                "", xy=(x, 1.7), xytext=(tx, 1.0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
            )

    # Level 2: final reduction
    ax.add_patch(mpatches.FancyBboxPatch(
        (3.5, 3.2), 1.0, 0.8,
        boxstyle="round,pad=0.1", linewidth=1.2,
        edgecolor="black", facecolor=node_color, alpha=0.9,
    ))
    ax.text(4.0, 3.6, "h1-4", ha="center", va="center", fontsize=10,
            color="white", fontweight="bold")
    for lx in level1_xs:
        ax.annotate(
            "", xy=(4.0, 3.2), xytext=(lx, 2.5),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
        )

    ax.text(4.0, 4.5, "Training: parallel scan  O(log T)",
            ha="center", va="center", fontsize=11, fontstyle="italic")

    # ── Right panel: sequential recurrence (inference) ───────────────────────
    ax = axes[1]
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5)
    ax.axis("off")

    state_ys = 2.8
    input_ys = 1.2
    output_ys = 4.2
    xs = [1.5, 3.5, 5.5, 7.5]
    state_labels  = ["h0", "h1", "h2", "h3"]
    input_labels  = ["x1", "x2", "x3", "x4"]
    output_labels = ["y1", "y2", "y3", "y4"]

    # Draw state nodes
    for x, label in zip(xs, state_labels):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.45, state_ys - 0.4), 0.9, 0.8,
            boxstyle="round,pad=0.1", linewidth=1.2,
            edgecolor="black", facecolor=node_color, alpha=0.8,
        ))
        ax.text(x, state_ys, label, ha="center", va="center", fontsize=11,
                color="white", fontweight="bold")

    # Horizontal arrows between states
    for i in range(len(xs) - 1):
        ax.annotate(
            "", xy=(xs[i + 1] - 0.45, state_ys),
            xytext=(xs[i] + 0.45, state_ys),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        )

    # Input tokens + upward arrows
    for x, label in zip(xs[1:], input_labels):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.45, input_ys - 0.4), 0.9, 0.8,
            boxstyle="round,pad=0.1", linewidth=1.2,
            edgecolor="black", facecolor=token_color,
        ))
        ax.text(x, input_ys, label, ha="center", va="center", fontsize=11)
        ax.annotate(
            "", xy=(x, state_ys - 0.4), xytext=(x, input_ys + 0.4),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2),
        )

    # Output tokens + upward arrows from states
    for x, label in zip(xs[1:], output_labels):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - 0.45, output_ys - 0.4), 0.9, 0.8,
            boxstyle="round,pad=0.1", linewidth=1.2,
            edgecolor="black", facecolor=token_color,
        ))
        ax.text(x, output_ys, label, ha="center", va="center", fontsize=11)
        ax.annotate(
            "", xy=(x, output_ys - 0.4), xytext=(x, state_ys + 0.4),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2),
        )

    ax.text(4.5, 4.95, "Inference: sequential recurrence  O(T)",
            ha="center", va="center", fontsize=11, fontstyle="italic")

    fig.tight_layout()
    return _save(fig, output_path)


# ── Convenience: generate all figures ────────────────────────────────────────

def generate_all(
    checkpoints: dict,
    sweep_results: str,
    output_dir: str,
    eval_results: dict | None = None,
    device: str = "cpu",
) -> dict:
    """Generate all Post A and Post B figures.

    Args:
        checkpoints:  {'transformer': path, 'tcn': path, 'mamba': path}
        sweep_results: path to checkpoints/sweep_results.json
        output_dir:   root output directory (figures/)
        eval_results: optional pre-computed eval dicts keyed by arch
        device:       torch device string

    Returns:
        dict of figure name -> saved path
    """
    import json

    with open(sweep_results) as f:
        sweeps = json.load(f)

    out = {}
    post_a = Path(output_dir) / "post_A"
    post_b = Path(output_dir) / "post_B"

    out["length_sweep_transformer"] = fig_length_sweep_transformer(
        sweeps["transformer"],
        str(post_a / "length_sweep_transformer.png"),
    )
    out["tcn_vs_transformer"] = fig_tcn_vs_transformer(
        sweeps["transformer"],
        sweeps["tcn"],
        str(post_a / "tcn_vs_transformer.png"),
        eval_results=eval_results,
    )
    out["all_three_sweep"] = fig_all_three_sweep(
        sweeps["transformer"],
        sweeps["tcn"],
        sweeps["mamba"],
        str(post_b / "all_three_sweep.png"),
    )
    if eval_results:
        out["perplexity_comparison"] = fig_perplexity_comparison(
            eval_results,
            str(post_b / "perplexity_comparison.png"),
        )
    out["training_inference_schematic"] = fig_training_inference_schematic(
        str(post_b / "training_inference_schematic.png"),
    )
    return out
