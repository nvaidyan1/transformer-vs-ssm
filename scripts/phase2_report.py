"""
plan2.md Phase 2 report, from saved JSON only.

2.1 (cost_scaling.json): log-log slopes of activation memory and time vs length, over the
full range and over the top three lengths; gate = every predicted top-three slope within
0.25 of its prediction. 2.2 (streaming_state.json): gate = analytic and measured carried
state agree within 5% at every context. Writes report.md, summary.json, cost_scaling.png
and streaming_state.png.

Usage:
    python scripts/phase2_report.py [results/phase2]
"""

import json
import math
import sys
from pathlib import Path

TOL_SLOPE = 0.25
TOL_STATE = 0.05
# (arm, metric) -> predicted top-three slope (plan2.md 2.1). Others are reported only.
PREDICTED = {
    ("transformer_naive", "memory"): 2.0,
    ("transformer_sdpa", "memory"): 1.0,
    ("transformer_sdpa", "time"): 2.0,
    ("tcn", "memory"): 1.0, ("tcn", "time"): 1.0,
    ("mamba_ssm", "memory"): 1.0, ("mamba_ssm", "time"): 1.0,
}
LABELS = {"transformer_naive": "Transformer, naive attention",
          "transformer_sdpa": "Transformer, SDPA", "tcn": "TCN", "mamba_ssm": "Mamba (mamba-ssm)",
          "transformer": "Transformer KV cache", "mamba": "Mamba state"}
COLORS = {"transformer_naive": "#2a78d6", "transformer_sdpa": "#1baf7a", "tcn": "#eb6834",
          "mamba_ssm": "#e87ba4", "transformer": "#2a78d6", "mamba": "#e87ba4"}


def slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y]
    if len(pts) < 2:
        return None
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    num = sum((a - mx) * (b - my) for a, b in pts)
    den = sum((a - mx) ** 2 for a, _ in pts)
    return num / den


def slopes(rows):
    L = [r["length"] for r in rows]
    mem = [r.get("activation_bytes") for r in rows]
    tim = [r.get("time_s") for r in rows]
    ok = [i for i in range(len(rows)) if mem[i] and tim[i]]
    top = ok[-3:]
    return {"memory_full": slope([L[i] for i in ok], [mem[i] for i in ok]),
            "memory_top3": slope([L[i] for i in top], [mem[i] for i in top]),
            "time_full": slope([L[i] for i in ok], [tim[i] for i in ok]),
            "time_top3": slope([L[i] for i in top], [tim[i] for i in top]),
            "top3_lengths": [L[i] for i in top], "oom_lengths": [L[i] for i in range(len(rows)) if i not in ok]}


def gate_21(all_slopes: dict) -> dict:
    checks = []
    for (arm, metric), pred in PREDICTED.items():
        s = (all_slopes.get(arm) or {}).get(f"{metric}_top3")
        checks.append({"arm": arm, "metric": metric, "predicted": pred, "measured": s,
                       "ok": s is not None and abs(s - pred) <= TOL_SLOPE})
    return {"verdict": "PASS" if all(c["ok"] for c in checks) else "FAIL", "checks": checks}


def gate_22(rec: dict) -> dict:
    worst = 0.0
    for a in rec["archs"].values():
        for m, an in zip(a["measured_bytes"], a["analytic_bytes"]):
            worst = max(worst, abs(m - an) / an if an else (0.0 if m == 0 else 1.0))
    return {"verdict": "PASS" if worst <= TOL_STATE else "FAIL", "worst_rel_diff": worst}


def plot_21(rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for arm, d in rec["arms"].items():
        rows = [r for r in d["rows"] if r.get("time_s")]
        L = [r["length"] for r in rows]
        a1.plot(L, [r["activation_bytes"] / 2**20 for r in rows], marker="o", ms=5, lw=2,
                color=COLORS[arm], label=LABELS[arm])
        a2.plot(L, [r["time_s"] * 1000 for r in rows], marker="o", ms=5, lw=2,
                color=COLORS[arm], label=LABELS[arm])
    for ax, yl in ((a1, "peak activation memory (MiB)"), (a2, "forward+backward time (ms)")):
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("sequence length"); ax.set_ylabel(yl)
        ax.grid(color="#e6e6e6", lw=0.6, which="both")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    a1.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(f"Training cost vs length, batch {rec['batch']}, {rec['gpu']} (~450k-param models)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_22(rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for arch, d in rec["archs"].items():
        ax.plot(rec["contexts"], [b / 1024 for b in d["measured_bytes"]], marker="o", ms=5, lw=2,
                color=COLORS[arch], label=LABELS.get(arch, arch))
    tcn = rec["archs"].get("tcn")
    if tcn and "max_window" not in tcn:
        # derived from the config, never written back into the raw record
        import yaml
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.models import build_model
        c = yaml.safe_load(open(Path(__file__).parent.parent / tcn["config"]))
        tcn = dict(tcn, max_window=build_model("tcn", vocab_size=c["vocab_size"],
                                               **c["model"]).max_padding)
    if tcn:
        # state saturates where the context covers the largest single layer's window
        # ((kernel - 1) * max dilation), which is smaller than the receptive field (the
        # sum over layers)
        sat = next((c for c, b in zip(rec["contexts"], tcn["measured_bytes"])
                    if b == tcn["measured_bytes"][-1]), None)
        for x, lab, y in ((tcn.get("max_window"), "largest TCN\nlayer window", 0.62),
                          (tcn.get("receptive_field"), "TCN receptive\nfield", 0.45)):
            if x:
                ax.axvline(x, color="#999", lw=1, ls="--")
                ax.text(x * 1.04, y, f"{lab} {x}", fontsize=7, color="#777",
                        transform=ax.get_xaxis_transform())
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("context length (tokens already generated)")
    ax.set_ylabel("state carried to the next token (KiB)")
    ax.grid(color="#e6e6e6", lw=0.6, which="both")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase2")
    out, lines = {}, []
    p21 = d / "cost_scaling.json"
    if p21.exists():
        rec = json.load(open(p21))
        s = {arm: slopes(v["rows"]) for arm, v in rec["arms"].items()}
        g = gate_21(s)
        out["2.1"] = {"slopes": s, "gate": g}
        lines += ["## 2.1 Compute and memory scaling", "",
                  "| model | params | memory slope full / top-3 | time slope full / top-3 | OOM at |",
                  "|---|---|---|---|---|"]
        f = lambda v: "n/a" if v is None else f"{v:.2f}"
        for arm, v in s.items():
            lines.append(f"| {LABELS[arm]} | {rec['arms'][arm]['n_params']:,} | {f(v['memory_full'])} / "
                         f"{f(v['memory_top3'])} | {f(v['time_full'])} / {f(v['time_top3'])} "
                         f"| {v['oom_lengths'] or '-'} |")
        lines.append(f"\nGate (top-3 slopes within {TOL_SLOPE} of prediction): **{g['verdict']}**")
        for c in g["checks"]:
            lines.append(f"- {LABELS[c['arm']]} {c['metric']}: predicted {c['predicted']}, "
                         f"measured {f(c['measured'])} {'ok' if c['ok'] else 'MISS'}")
        plot_21(rec, d / "cost_scaling.png")
    p22 = d / "streaming_state.json"
    if p22.exists():
        rec = json.load(open(p22))
        g = gate_22(rec)
        out["2.2"] = {"gate": g}
        lines += ["", "## 2.2 Streaming inference state (bytes carried to the next token)", "",
                  "| context | " + " | ".join(LABELS.get(a, a) for a in rec["archs"]) + " |",
                  "|---|" + "---|" * len(rec["archs"])]
        for i, c in enumerate(rec["contexts"]):
            lines.append(f"| {c} | " + " | ".join(f"{a['measured_bytes'][i]:,}" for a in rec["archs"].values()) + " |")
        lines.append(f"\nGate (analytic vs measured within 5%): **{g['verdict']}** "
                     f"(worst relative difference {g['worst_rel_diff']:.2%})")
        plot_22(rec, d / "streaming_state.png")
    json.dump(out, open(d / "summary.json", "w"), indent=2)
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
