"""
plan2.md 3.2 report, from saved JSON only: accuracy vs min_gap per model (3-seed mean,
min, max), the gate, the D3 extension list, and accuracy_vs_gap.png with the receptive
fields marked.

Usage:
    python scripts/phase3_report.py [results/phase3]
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from cap_rule import needs_extension  # noqa: E402
from src.tasks import chance_accuracy  # noqa: E402

ARMS = ("tcn_short", "tcn_small", "transformer", "mamba_ssm")
RF = {"tcn_short": 127, "tcn_small": 769}
LABELS = {"tcn_short": "TCN, RF 127", "tcn_small": "TCN, RF 769",
          "transformer": "Transformer", "mamba_ssm": "Mamba (mamba-ssm)"}
COLORS = {"tcn_short": "#eb6834", "tcn_small": "#eda100", "transformer": "#2a78d6",
          "mamba_ssm": "#e87ba4"}
SMALLEST = 32


def load(d: Path) -> dict:
    """(arm, gap) -> [records], preferring an extended run (dir ext/) over its CAP run."""
    out = {}
    for p in sorted(glob.glob(str(d / "*_g*_s*.json"))):
        r = json.load(open(p))
        arm, gap = Path(p).stem.rsplit("_g", 1)[0], int(Path(p).stem.split("_g")[-1].split("_s")[0])
        ext = d / "ext" / Path(p).name
        if ext.exists():
            r = json.load(open(ext))
        out.setdefault((arm, gap), []).append(r)
    return out


def stats(rs):
    a = [r["history"][-1]["val_acc"] for r in rs if r.get("complete", True)]
    return {"n": len(a), "mean": sum(a) / len(a) if a else None,
            "min": min(a) if a else None, "max": max(a) if a else None}


def gate(cells: dict) -> dict:
    chance = chance_accuracy("selective_copy", 64)
    above = sorted(g for (a, g) in cells if a == "tcn_short" and g > RF["tcn_short"])
    checks = []
    for g in above:
        s = stats(cells[("tcn_short", g)])
        checks.append({"gap": g, "mean": s["mean"], "n": s["n"],
                       "ok": s["n"] == 3 and s["mean"] <= 2 * chance})
    s = stats(cells.get(("tcn_short", SMALLEST), []))
    checks.append({"gap": SMALLEST, "mean": s["mean"], "n": s["n"],
                   "ok": s["n"] == 3 and s["mean"] > 0.9})
    complete = all(c["n"] == 3 for c in checks) and len(above) == 3
    return {"verdict": ("PASS" if all(c["ok"] for c in checks) else "FAIL") if complete
            else "INCOMPLETE", "checks": checks}


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase3")
    cells = load(d)
    lines = ["| model | min_gap | seeds | final val_acc mean [min, max] |", "|---|---|---|---|"]
    for arm in ARMS:
        for g in sorted(g for (a, g) in cells if a == arm):
            s = stats(cells[(arm, g)])
            lines.append(f"| {LABELS[arm]} | {g} | {s['n']} | {s['mean']:.3f} [{s['min']:.3f}, {s['max']:.3f}] |"
                         if s["n"] else f"| {LABELS[arm]} | {g} | 0 | not run |")
    g = gate(cells)
    lines.append(f"\nGate (tcn_short <= 2x chance at every gap above RF 127, > 0.9 at gap 32): **{g['verdict']}**")
    ext = []
    for p in sorted(glob.glob(str(d / "*_g*_s*.json"))):
        r = json.load(open(p))
        why = needs_extension(r) if not (d / "ext" / Path(p).name).exists() else []
        if why:
            ext.append({"run": Path(p).stem, "reasons": why})
    lines.append(f"\nD3 extensions needed: {len(ext)}")
    lines += [f"- {e['run']}: {'; '.join(e['reasons'])}" for e in ext]
    json.dump({"gate": g, "extensions": ext}, open(d / "summary.json", "w"), indent=2)
    if cells:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for arm in ARMS:
            gs = sorted(g for (a, g) in cells if a == arm and stats(cells[(a, g)])["n"])
            if not gs:
                continue
            st = [stats(cells[(arm, g)]) for g in gs]
            ax.errorbar(gs, [s["mean"] for s in st],
                        yerr=[[s["mean"] - s["min"] for s in st], [s["max"] - s["mean"] for s in st]],
                        color=COLORS[arm], lw=2, marker="o", ms=7, capsize=3, label=LABELS[arm])
        for arm, rf in RF.items():
            ax.axvline(rf, color=COLORS[arm], lw=1, ls="--")
            ax.text(rf * 1.04, 0.5, f"RF {rf}", fontsize=7, color="#777",
                    transform=ax.get_xaxis_transform())
        ax.axhline(chance_accuracy("selective_copy", 64), color="#999", lw=1, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("min_gap: every answer is at least this far from its source")
        ax.set_ylabel("final val_acc (mean, min-max of 3 seeds)")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(color="#e6e6e6", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
        fig.tight_layout(); fig.savefig(d / "accuracy_vs_gap.png", dpi=150); plt.close(fig)
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
