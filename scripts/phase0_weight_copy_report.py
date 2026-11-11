"""
Report for the weight-copied pair (plan2.md 0.2 follow-up), from saved JSON only.
Writes table.md (both accuracy curves and the paired difference at every eval) and
curves.png next to the record.

Usage:
    python scripts/phase0_weight_copy_report.py results/phase0_weight_copy/pair_emb1.0_d1d0_s0.json
"""

import json
import sys
from pathlib import Path

COLORS = {"official": "#2a78d6", "custom": "#eb6834"}   # categorical slots 1, 2


def rows(record: dict) -> list:
    return [{"step": h["step"], "official": h["official"]["val_acc"],
             "custom": h["custom"]["val_acc"],
             "diff": h["custom"]["val_acc"] - h["official"]["val_acc"]}
            for h in record["history"]]


def to_markdown(record: dict) -> str:
    lines = [f"Weight-copied pair: emb_init_std={record['emb_init_std']}, "
             f"decay_1d={record['decay_1d']}, seed={record['seed']}, "
             f"complete={record['complete']}. Step-0 max |logit diff| after copy: "
             f"{record['init_max_logit_diff']:.2e}.",
             "", "| step | official val_acc | custom val_acc | custom - official |",
             "|---|---|---|---|"]
    for r in rows(record):
        lines.append(f"| {r['step']} | {r['official']:.4f} | {r['custom']:.4f} "
                     f"| {r['diff']:+.4f} |")
    return "\n".join(lines)


def plot(record: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = rows(record)
    steps = [r["step"] for r in rs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in ("official", "custom"):
        ys = [r[k] for r in rs]
        a1.plot(steps, ys, color=COLORS[k], lw=2, marker="o", ms=4, label=k)
        a1.annotate(k, (steps[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="#555", va="center")
    a1.axhline(record["chance_acc"], color="#999", lw=1, ls="--")
    a1.set_title("val_acc, weight-copied start", fontsize=9)
    a1.set_ylabel("val_acc (answer positions)")
    a1.legend(fontsize=8, frameon=False, loc="upper left")
    a2.axhline(0, color="#999", lw=1)
    a2.plot(steps, [r["diff"] for r in rs], color="#555", lw=2, marker="o", ms=4)
    a2.set_title("paired difference (custom - official)", fontsize=9)
    for ax in (a1, a2):
        ax.set_xlabel("step")
        ax.grid(color="#e6e6e6", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"emb_init_std={record['emb_init_std']}, decay_1d={record['decay_1d']}, "
                 f"seed {record['seed']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = Path(sys.argv[1])
    record = json.load(open(p))
    md = to_markdown(record)
    (p.parent / "table.md").write_text(md + "\n")
    plot(record, p.parent / "curves.png")
    print(md)


if __name__ == "__main__":
    main()
