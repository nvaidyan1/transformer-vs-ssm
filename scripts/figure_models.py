"""
Figures from the saved Selective Copying models (results/figure_models; CLAIM_MAP rows
Post 1 Figs 3-4 and Post 2 Fig 4). Illustrations, not reported numbers. CPU.

attention_maps.png  the trained Transformer's attention (heads averaged), every layer, on
                    two inputs whose data tokens sit at different positions: the routing
                    is recomputed per input.
delta_trace.png     Delta (per channel, one layer) across one input for the trained
                    mamba_ssm model, run through the custom model after copying its
                    weights (names map blocks.i.mixer.* -> blocks.i.ssm.*). The copy is
                    checked by re-measuring accuracy on the validation set.

Usage:
    python scripts/figure_models.py [results/figure_models]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.models import build_model
from src.tasks import make_batcher
from src.train_task import VAL_SEED_BASE


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def accuracy(model, batcher, n=4):
    c = t = 0
    with torch.no_grad():
        for i in range(n):
            tok, tgt, mask, _ = batcher(32, VAL_SEED_BASE + i)
            pred = model(torch.from_numpy(tok)).argmax(-1).numpy()
            c += int((pred[mask] == tgt[mask]).sum()); t += int(mask.sum())
    return c / t


def two_inputs(batcher):
    """Two validation sequences whose data positions differ as much as possible."""
    tok, _, _, metas = batcher(64, VAL_SEED_BASE + 99)
    first = [m["source_positions"][0] for m in metas]
    i, j = int(np.argmin(first)), int(np.argmax(first))
    return tok[[i, j]], [metas[i], metas[j]]


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/figure_models")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}

    ck = load(d / "transformer_s0.pt")
    tf = build_model("transformer", vocab_size=ck["vocab_size"], **ck["model_cfg"]).eval()
    tf.load_state_dict(ck["state_dict"])
    batcher = make_batcher("selective_copy", **ck["task_args"])
    out["transformer_val_acc"] = accuracy(tf, batcher)
    tok, metas = two_inputs(batcher)
    with torch.no_grad():
        _, attn = tf(torch.from_numpy(tok), return_attention=True)
    L = len(attn)
    fig, axes = plt.subplots(2, L, figsize=(3.4 * L, 6.6))
    for row in range(2):
        m = metas[row]
        for l in range(L):
            ax = axes[row, l]
            a = attn[l][row].mean(0).numpy()                    # heads averaged, (T, T)
            ax.imshow(a, cmap="Blues", vmin=0, vmax=a.max())
            for p in m["source_positions"]:
                ax.axvline(p, color="#eb6834", lw=0.6, alpha=0.7)
            ax.axhline(m["marker_position"], color="#999", lw=0.6, ls="--")
            ax.set_title(f"input {row + 1}, layer {l + 1}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].set_xlabel("attended position (orange: data tokens)", fontsize=7)
    axes[1, 0].set_ylabel("query position (dashed: marker)", fontsize=7)
    fig.suptitle("Trained Transformer on Selective Copying: attention, heads averaged",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(d / "attention_maps.png", dpi=150); plt.close(fig)

    ck = load(d / "mamba_ssm_s0.pt")
    cfg = {k: v for k, v in ck["model_cfg"].items() if k != "use_fast_path"}
    mb = build_model("mamba", vocab_size=ck["vocab_size"], scan="sequential",
                     grad_checkpoint=False, **cfg).eval()
    sd = {k.replace(".mixer.", ".ssm."): v for k, v in ck["state_dict"].items()}
    missing, unexpected = mb.load_state_dict(sd, strict=False)
    assert not unexpected and all(".fixed_" in k for k in missing), (missing, unexpected)
    out["mamba_copied_val_acc"] = accuracy(mb, make_batcher("selective_copy", **ck["task_args"]))
    with torch.no_grad():
        _, deltas = mb(torch.from_numpy(tok[:1]), return_delta=True)
    layer = int(np.argmax([float(dl[0].std(0).mean()) for dl in deltas]))   # most varied layer
    D = deltas[layer][0].numpy()                                            # (T, E)
    m = metas[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3]})
    a1.imshow(D.T, aspect="auto", cmap="Blues", interpolation="nearest")
    a1.set_ylabel(f"channel (layer {layer + 1})")
    a2.plot(np.arange(D.shape[0]), D.mean(1), color="#2a78d6", lw=1.5, label="mean over channels")
    for ax in (a1, a2):
        for p in m["source_positions"]:
            ax.axvline(p, color="#eb6834", lw=0.8, alpha=0.8)
        ax.axvline(m["marker_position"], color="#999", lw=0.8, ls="--")
    a2.set_xlabel("position (orange: data tokens; dashed: marker)")
    a2.set_ylabel("Delta"); a2.legend(fontsize=7, frameon=False)
    fig.suptitle("Trained Mamba on Selective Copying: Delta per channel (one of several "
                 "input-dependent quantities, with B_t, C_t and the gate)", fontsize=9)
    fig.tight_layout(); fig.savefig(d / "delta_trace.png", dpi=150); plt.close(fig)
    out["delta_layer_shown"] = layer + 1
    json.dump(out, open(d / "figures_summary.json", "w"), indent=2)
    print(out)


if __name__ == "__main__":
    main()
