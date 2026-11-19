"""
plan2.md 4.1 report, from saved JSON only: final accuracy vs number of KV pairs per model
(3-seed mean, min, max), the gate, the D3 extension list, and accuracy_vs_pairs.png (one
line per model size; Mamba solid, Transformer dashed).

Usage:
    python scripts/phase4_report.py [results/phase4]
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from cap_rule import needs_extension  # noqa: E402

SIZES = (32, 64, 128)
PAIRS = (4, 8, 16, 32, 64)
SIZE_COLORS = {32: "#86b6ef", 64: "#2a78d6", 128: "#104281"}   # one-hue ramp by size


def load(d: Path) -> dict:
    out = {}
    for p in sorted(glob.glob(str(d / "*_p*_s*.json"))):
        stem = Path(p).stem
        model, rest = stem.rsplit("_p", 1)
        pairs = int(rest.split("_s")[0])
        ext = d / "ext" / Path(p).name
        r = json.load(open(ext if ext.exists() else p))
        out.setdefault((model, pairs), []).append(r)
    return out


def mean_acc(rs):
    a = [r["history"][-1]["val_acc"] for r in rs if r.get("complete", True)]
    return (sum(a) / len(a), min(a), max(a), len(a)) if a else (None, None, None, 0)


GATE_CELLS = (("mamba_ssm_d32", 4), ("mamba_ssm_d32", 64),
              ("transformer_m32", 4), ("transformer_m32", 64))


def gate(cells: dict, pending: set = frozenset()) -> dict:
    """pending: (model, pairs) cells that still need a D3 extension; the gate is only
    evaluated after extensions (results/phase4/PREDICTION.md)."""
    waiting = [c for c in GATE_CELLS if c in pending]
    if waiting:
        return {"verdict": "INCOMPLETE", "awaiting_extension": [f"{m} {p} pairs" for m, p in waiting]}
    m4, _, _, n1 = mean_acc(cells.get(("mamba_ssm_d32", 4), []))
    m64, _, _, n2 = mean_acc(cells.get(("mamba_ssm_d32", 64), []))
    t4, _, _, n3 = mean_acc(cells.get(("transformer_m32", 4), []))
    t64, _, _, n4 = mean_acc(cells.get(("transformer_m32", 64), []))
    if min(n1, n2, n3, n4) < 3:
        return {"verdict": "INCOMPLETE"}
    ok = (m4 - m64) >= 0.3 and (t4 - t64) < 0.1
    return {"verdict": "PASS" if ok else "FAIL", "mamba_drop": m4 - m64,
            "transformer_drop": t4 - t64}


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase4")
    cells = load(d)
    models = [f"mamba_ssm_d{s}" for s in SIZES] + [f"transformer_m{s}" for s in SIZES]
    lines = ["| model | " + " | ".join(f"{p} pairs" for p in PAIRS) + " |",
             "|---|" + "---|" * len(PAIRS)]
    for m in models:
        row = []
        for p in PAIRS:
            mu, lo, hi, n = mean_acc(cells.get((m, p), []))
            row.append(f"{mu:.3f} [{lo:.2f}, {hi:.2f}]" + ("" if n == 3 else f" (n={n})")
                       if n else "-")
        lines.append(f"| {m} | " + " | ".join(row) + " |")
    ext = []
    for p in sorted(glob.glob(str(d / "*_p*_s*.json"))):
        if (d / "ext" / Path(p).name).exists():
            continue
        why = needs_extension(json.load(open(p)))
        if why:
            ext.append({"run": Path(p).stem, "reasons": why})
    pending = {(e["run"].rsplit("_p", 1)[0], int(e["run"].rsplit("_p", 1)[1].split("_s")[0])) for e in ext}
    g = gate(cells, pending)
    lines.append(f"\nGate (mamba_ssm_d32 drops >= 0.3 from 4 to 64 pairs; transformer_m32 "
                 f"drops < 0.1): **{g['verdict']}**" +
                 (f" (Mamba drop {g['mamba_drop']:+.3f}, Transformer drop "
                  f"{g['transformer_drop']:+.3f})" if "mamba_drop" in g else "")
                 + (f" (awaiting D3 extension: {', '.join(g['awaiting_extension'])})"
                    if "awaiting_extension" in g else ""))
    lines.append(f"\nD3 extensions needed: {len(ext)}")
    lines += [f"- {e['run']}: {'; '.join(e['reasons'])}" for e in ext]
    json.dump({"gate": g, "extensions": ext}, open(d / "summary.json", "w"), indent=2)
    if cells:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for s in SIZES:
            for m, ls, lab in ((f"mamba_ssm_d{s}", "-", f"Mamba d_model {s}"),
                               (f"transformer_m{s}", "--", f"Transformer matched to d{s}")):
                ps = [p for p in PAIRS if mean_acc(cells.get((m, p), []))[3]]
                if ps:
                    ax.plot(ps, [mean_acc(cells[(m, p)])[0] for p in ps], ls=ls, lw=2,
                            marker="o", ms=6, color=SIZE_COLORS[s], label=lab)
        ax.axhline(1 / 127, color="#999", lw=1, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xticks(PAIRS); ax.set_xticklabels(PAIRS)
        ax.set_xlabel("key-value pairs to recall (MQAR)")
        ax.set_ylabel("final val_acc (mean of 3 seeds)")
        ax.set_ylim(-0.03, 1.05)
        ax.grid(color="#e6e6e6", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
        fig.tight_layout(); fig.savefig(d / "accuracy_vs_pairs.png", dpi=150); plt.close(fig)
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
