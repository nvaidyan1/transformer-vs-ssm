"""
plan2.md 0.2 report: table, gate verdict and learning curves, from saved JSON only.

Reads every run record written by src/train_task.py under a results directory
(default results/phase0_2x2/), never touches a model. Writes, next to them:
    summary.json   per-cell mean/min/max val_acc at steps 1000 and 2000, and the gate
    table.md       the same as a markdown table
    curves.png     val_acc vs step, one panel per cell, custom vs official

Usage:
    python scripts/phase0_2x2_report.py [results/phase0_2x2]
"""

import json
import sys
from pathlib import Path

REPORT_STEPS = (1000, 2000)
GATE_STEP = 2000
GATE_MEAN_TOL = 0.1
N_SEEDS_REQUIRED = 3
ARCHS = ("mamba", "mamba_ssm")          # custom, official
LABELS = {"mamba": "custom", "mamba_ssm": "official"}
COLORS = {"mamba_ssm": "#2a78d6", "mamba": "#eb6834"}   # categorical slots 1, 2


def load_records(results_dir: Path) -> list:
    recs = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name in ("summary.json", "manifest.json"):
            continue
        r = json.load(open(p))
        if "history" in r and "arch" in r:
            recs.append(r)
    return recs


def acc_at(record: dict, step: int):
    for h in record["history"]:
        if h["step"] == step:
            return h["val_acc"]
    return None


def cell_key(record: dict) -> tuple:
    return (record["emb_init_std"], bool(record["decay_1d"]))


def stats(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "min": None, "max": None}
    return {"n": len(vals), "mean": sum(vals) / len(vals),
            "min": min(vals), "max": max(vals)}


def cell_gate(custom: dict, official: dict) -> dict:
    """Gate for one cell from the step-2000 stats of each arch."""
    if custom["n"] < N_SEEDS_REQUIRED or official["n"] < N_SEEDS_REQUIRED:
        return {"verdict": "INCOMPLETE", "overlap": None, "mean_diff": None}
    overlap = custom["min"] <= official["max"] and official["min"] <= custom["max"]
    diff = custom["mean"] - official["mean"]
    ok = overlap or abs(diff) <= GATE_MEAN_TOL
    return {"verdict": "PASS" if ok else "FAIL", "overlap": overlap, "mean_diff": diff}


def summarize(records: list) -> dict:
    cells = {}
    for r in records:
        if r["arch"] not in ARCHS:
            continue
        cells.setdefault(cell_key(r), {a: [] for a in ARCHS})[r["arch"]].append(r)

    out = []
    for (emb, d1d), by_arch in sorted(cells.items()):
        row = {"emb_init_std": emb, "decay_1d": d1d}
        for arch in ARCHS:
            row[arch] = {f"step{s}": stats([acc_at(r, s) for r in by_arch[arch]])
                         for s in REPORT_STEPS}
            row[arch]["seeds"] = sorted(r["seed"] for r in by_arch[arch])
        row["gate"] = cell_gate(row["mamba"][f"step{GATE_STEP}"],
                                row["mamba_ssm"][f"step{GATE_STEP}"])
        out.append(row)

    verdicts = [c["gate"]["verdict"] for c in out]
    if len(out) < 4 or "INCOMPLETE" in verdicts:
        overall = "INCOMPLETE"
    else:
        overall = "PASS" if all(v == "PASS" for v in verdicts) else "FAIL"
    return {"cells": out, "gate": overall, "gate_step": GATE_STEP,
            "gate_mean_tol": GATE_MEAN_TOL}


def fmt(s: dict) -> str:
    if not s["n"]:
        return "n/a"
    return f"{s['mean']:.3f} [{s['min']:.3f}, {s['max']:.3f}]"


def to_markdown(summary: dict) -> str:
    lines = ["| emb_init_std | decay_1d | arch | n | val_acc @1000 mean [min, max] "
             "| val_acc @2000 mean [min, max] | gate |",
             "|---|---|---|---|---|---|---|"]
    for c in summary["cells"]:
        g = c["gate"]
        gate = g["verdict"]
        if g["mean_diff"] is not None:
            gate += f" (overlap={g['overlap']}, diff={g['mean_diff']:+.3f})"
        for i, arch in enumerate(ARCHS):
            s = c[arch]
            lines.append(f"| {c['emb_init_std']} | {c['decay_1d']} | {LABELS[arch]} "
                         f"| {s['step2000']['n']} | {fmt(s['step1000'])} "
                         f"| {fmt(s['step2000'])} | {gate if i == 0 else ''} |")
    lines.append(f"\nOverall gate: **{summary['gate']}**")
    return "\n".join(lines)


def plot(records: list, summary: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [(c["emb_init_std"], c["decay_1d"]) for c in summary["cells"]]
    n = max(1, len(keys))
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows),
                             sharex=True, sharey=True, squeeze=False)
    chance = records[0]["chance_acc"] if records else None
    for ax, key, cell in zip(axes.flat, keys, summary["cells"]):
        for arch in ARCHS:
            runs = [r for r in records if r["arch"] == arch and cell_key(r) == key]
            for i, r in enumerate(sorted(runs, key=lambda r: r["seed"])):
                steps = [h["step"] for h in r["history"]]
                accs = [h["val_acc"] for h in r["history"]]
                ax.plot(steps, accs, color=COLORS[arch], lw=2, alpha=0.85,
                        label=f"{LABELS[arch]} (3 seeds)" if i == 0 else None)
            if runs:
                last = max(runs, key=lambda r: r["history"][-1]["val_acc"])
                h = last["history"][-1]
                ax.annotate(LABELS[arch], (h["step"], h["val_acc"]),
                            xytext=(4, 0), textcoords="offset points",
                            fontsize=8, color="#555", va="center")
        if chance:
            ax.axhline(chance, color="#999", lw=1, ls="--")
            ax.text(20, chance, "chance", fontsize=7, color="#777", va="bottom")
        g = cell["gate"]["verdict"]
        ax.set_title(f"emb_init_std={key[0]}, decay_1d={key[1]}  [{g}]", fontsize=9)
        ax.grid(color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("step")
    for ax in axes[:, 0]:
        ax.set_ylabel("val_acc (answer positions)")
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Selective Copying (n_data=4, seq_len=64): custom vs official Mamba",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase0_2x2")
    records = load_records(results_dir)
    summary = summarize(records)
    json.dump(summary, open(results_dir / "summary.json", "w"), indent=2)
    md = to_markdown(summary)
    (results_dir / "table.md").write_text(md + "\n")
    if records:
        plot(records, summary, results_dir / "curves.png")
    print(md)


if __name__ == "__main__":
    main()
