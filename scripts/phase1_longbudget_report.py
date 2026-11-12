"""
Schedule follow-up (c) report, from saved JSON only: ablated mamba_ssm, n_data 4, random
positions, seed 0, 20,000 steps, at seq_len 64 and 256. Writes report.md and curves.png.

Usage:
    python scripts/phase1_longbudget_report.py [results/phase1_sched_followups/longbudget]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.tasks import record_chance  # noqa: E402

RUNS = (("ablated_nd4_seq64_s0", "seq_len 64", "#2a78d6"),
        ("ablated_nd4_seq256_s0", "seq_len 256", "#eb6834"))


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1_sched_followups/longbudget")
    recs = {n: json.load(open(d / f"{n}.json")) for n, _, _ in RUNS if (d / f"{n}.json").exists()}
    lines = ["| run | steps run | complete | final val_acc | max val_acc | below 0.2 throughout |",
             "|---|---|---|---|---|---|"]
    for n, label, _ in RUNS:
        r = recs.get(n)
        if not r:
            lines.append(f"| {label} | not run | | | | |")
            continue
        accs = [h["val_acc"] for h in r["history"]]
        lines.append(f"| {label} | {r['steps_run']} | {r.get('complete', True)} | {accs[-1]:.4f} "
                     f"| {max(accs):.4f} | {max(accs) < 0.2} |")
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)
    if recs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3.6))
        for n, label, c in RUNS:
            if n in recs:
                h = recs[n]["history"]
                ax.plot([x["step"] for x in h], [x["val_acc"] for x in h], color=c, lw=2,
                        label=f"ablated, n_data 4, {label}")
        ax.axhline(0.2, color="#999", lw=1)
        ax.text(20200, 0.2, "0.2", fontsize=7, color="#777", va="center")
        ax.axhline(record_chance(next(iter(recs.values()))), color="#999", lw=1, ls="--")
        ax.set_xlabel("step (20,000-step schedule, no early stop)")
        ax.set_ylabel("val_acc")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(color="#e6e6e6", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(d / "curves.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
