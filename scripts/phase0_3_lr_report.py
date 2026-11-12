"""
plan2.md 0.3 learning-rate check report, from saved JSON only. Applies the
instability and selection rules fixed in results/phase0_3_lr/PREDICTION.md.
Writes summary.json, table.md and curves.png next to the records.

Usage:
    python scripts/phase0_3_lr_report.py [results/phase0_3_lr]
"""

import json
import math
import sys
from pathlib import Path

FINAL_STEP = 4000
COLLAPSE_TOL = 0.05
TIE_TOL = 0.005
ARCHS = ("transformer", "tcn", "mamba_ssm")
# lr is ordinal: one-hue sequential ramp, light (low lr) -> dark (high lr)
LR_COLORS = ["#86b6ef", "#2a78d6", "#104281"]


def load_records(d: Path) -> list:
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name == "summary.json" or p.name.startswith("manifest"):
            continue
        r = json.load(open(p))
        if "history" in r and "arch" in r:
            out.append(r)
    return out


def instability(record: dict) -> list:
    """Reasons this run is unstable or not selectable (empty list = stable)."""
    h = record["history"]
    if not record.get("complete", True) or not h or h[-1]["step"] != FINAL_STEP:
        return ["incomplete"]
    reasons = []
    if any(not math.isfinite(x[k]) for x in h for k in ("train_loss", "val_loss")):
        reasons.append("non-finite loss")
    final, peak = h[-1]["val_acc"], max(x["val_acc"] for x in h)
    if final < peak - COLLAPSE_TOL:
        reasons.append(f"collapse: final {final:.3f} < peak {peak:.3f} - {COLLAPSE_TOL}")
    if h[-1]["val_loss"] > h[0]["val_loss"]:
        reasons.append(f"val_loss rose: {h[0]['val_loss']:.3f} -> {h[-1]['val_loss']:.3f}")
    return reasons


def pick(records: list):
    """Per the selection rule: highest final val_acc among stable runs; within
    TIE_TOL of the best, the lowest lr. None if no stable run."""
    stable = [r for r in records if not instability(r)]
    if not stable:
        return None
    best = max(r["history"][-1]["val_acc"] for r in stable)
    near = [r for r in stable if r["history"][-1]["val_acc"] >= best - TIE_TOL]
    return min(near, key=lambda r: r["lr"])["lr"]


def summarize(records: list) -> dict:
    out = {}
    for arch in ARCHS:
        rs = sorted([r for r in records if r["arch"] == arch], key=lambda r: r["lr"])
        out[arch] = {
            "runs": [{"lr": r["lr"], "n_params": r["n_params"],
                      "final_val_acc": r["history"][-1]["val_acc"] if r["history"] else None,
                      "peak_val_acc": max((x["val_acc"] for x in r["history"]), default=None),
                      "steps": r["history"][-1]["step"] if r["history"] else 0,
                      "unstable": instability(r)} for r in rs],
            "picked_lr": pick(rs),
        }
    return out


def to_markdown(summary: dict) -> str:
    lines = ["| arch | params | lr | val_acc @4000 | peak val_acc | stability | picked |",
             "|---|---|---|---|---|---|---|"]
    for arch, s in summary.items():
        for r in s["runs"]:
            fin = f"{r['final_val_acc']:.4f}" if r["final_val_acc"] is not None else "n/a"
            peak = f"{r['peak_val_acc']:.4f}" if r["peak_val_acc"] is not None else "n/a"
            stab = "; ".join(r["unstable"]) or "stable"
            lines.append(f"| {arch} | {r['n_params']:,} | {r['lr']:g} | {fin} | {peak} "
                         f"| {stab} | {'**yes**' if r['lr'] == s['picked_lr'] else ''} |")
    lines.append("")
    for arch, s in summary.items():
        lines.append(f"- {arch}: picked lr = "
                     f"{s['picked_lr']:g}" if s["picked_lr"] is not None
                     else f"- {arch}: NO stable run, no lr picked")
    return "\n".join(lines)


def plot(records: list, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    lrs = sorted({r["lr"] for r in records})
    for ax, arch in zip(axes, ARCHS):
        for r in sorted([r for r in records if r["arch"] == arch], key=lambda r: r["lr"]):
            c = LR_COLORS[lrs.index(r["lr"]) % len(LR_COLORS)]
            ax.plot([h["step"] for h in r["history"]], [h["val_acc"] for h in r["history"]],
                    color=c, lw=2, ls="-" if r.get("complete", True) else "--",
                    label=f"lr {r['lr']:g}")
        if records:
            ax.axhline(records[0]["chance_acc"], color="#999", lw=1, ls="--")
        ax.set_title(arch, fontsize=9)
        ax.set_xlabel("step")
        ax.grid(color="#e6e6e6", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="center right")
    axes[0].set_ylabel("val_acc (answer positions)")
    fig.suptitle("Learning-rate check: Selective Copying (n_data=4, seq_len=64), seed 0",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase0_3_lr")
    records = load_records(d)
    summary = summarize(records)
    json.dump(summary, open(d / "summary.json", "w"), indent=2)
    md = to_markdown(summary)
    (d / "table.md").write_text(md + "\n")
    if records:
        plot(records, d / "curves.png")
    print(md)


if __name__ == "__main__":
    main()
