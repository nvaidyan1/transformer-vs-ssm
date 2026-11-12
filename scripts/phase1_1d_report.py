"""
plan2.md 1.1d report and Post 2 figure, from saved JSON only: final val_acc vs n_data
for mamba_ssm selective and ablated (random positions, seq_len 64), 3 seeds. No gate.
Writes report.md, summary.json and items_carried.png next to the records.

Usage:
    python scripts/phase1_1d_report.py [results/phase1_1d]
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tasks import chance_accuracy  # noqa: E402

N_DATA = (1, 2, 3, 4)
COLORS = {"selective": "#2a78d6", "ablated": "#eb6834"}   # categorical slots 1, 2


def collect(d: Path) -> dict:
    out = {}
    for arm in COLORS:
        for nd in N_DATA:
            rs = [json.load(open(p)) for p in sorted(glob.glob(str(d / f"{arm}_nd{nd}_s*.json")))]
            rs = [r for r in rs if r.get("complete", True)]
            accs = [r["history"][-1]["val_acc"] for r in rs]
            out[(arm, nd)] = {
                "n": len(accs), "accs": accs,
                "mean": sum(accs) / len(accs) if accs else None,
                "min": min(accs) if accs else None, "max": max(accs) if accs else None,
                "slot_acc_mean": ([sum(s) / len(rs) for s in zip(*[r["slot_acc"] for r in rs])]
                                  if rs and all("slot_acc" in r for r in rs) else None)}
    return out


def plot(stats: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for arm, c in COLORS.items():
        xs = [nd for nd in N_DATA if stats[(arm, nd)]["n"]]
        if not xs:
            continue
        mean = [stats[(arm, nd)]["mean"] for nd in xs]
        lo = [stats[(arm, nd)]["mean"] - stats[(arm, nd)]["min"] for nd in xs]
        hi = [stats[(arm, nd)]["max"] - stats[(arm, nd)]["mean"] for nd in xs]
        label = ("selective (input-dependent dynamics)" if arm == "selective"
                 else "ablated (input-independent dynamics)")
        ax.errorbar(xs, mean, yerr=[lo, hi], color=c, lw=2, marker="o", ms=8, capsize=4,
                    label=label)
    ax.axhline(chance_accuracy("selective_copy", 64), color="#999", lw=1, ls="--")
    ax.text(4.12, chance_accuracy("selective_copy", 64), "chance", fontsize=7, color="#777",
            va="center")
    ax.set_xticks(N_DATA)
    ax.set_xlim(0.7, 4.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("data tokens to carry (n_data), random positions, seq_len 64")
    ax.set_ylabel("val_acc at final eval (mean, min-max of 3 seeds)")
    ax.grid(color="#e6e6e6", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1_1d")
    stats = collect(d)
    lines = ["| arm | n_data | seeds | final val_acc mean [min, max] | per-slot mean |",
             "|---|---|---|---|---|"]
    for (arm, nd), s in stats.items():
        if not s["n"]:
            lines.append(f"| {arm} | {nd} | 0 | not run | |")
            continue
        slots = ", ".join(f"{a:.2f}" for a in s["slot_acc_mean"]) if s["slot_acc_mean"] else ""
        lines.append(f"| {arm} | {nd} | {s['n']} | {s['mean']:.3f} [{s['min']:.3f}, "
                     f"{s['max']:.3f}] | {slots} |")
    lines.append(f"\nChance {chance_accuracy('selective_copy', 64):.4f}. No gate.")
    json.dump({f"{a}_nd{n}": v for (a, n), v in stats.items()},
              open(d / "summary.json", "w"), indent=2)
    if any(s["n"] for s in stats.values()):
        plot(stats, d / "items_carried.png")
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
