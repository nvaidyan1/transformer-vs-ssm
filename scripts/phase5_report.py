"""
plan2.md 5.2 and 5.3 report, from saved JSON only.

5.2: final validation MSE per task x arm x depth (3-seed mean, min, max) against the exact
Kalman filter and the best constant-gain filter. Gates (1 layer): (a) stationary: ablation
mean within 10% of selective mean; (b) switching: ablation >= 15% worse than selective on
3/3 seeds, paired by seed. 5.3 (gain.json): Pearson r between selective Mamba's effective
gain and K_t > 0.7 on 3/3 seeds (1 layer). Figures: gain_scatter.png, delta_regime.png,
example_trace.png.

Usage:
    python scripts/phase5_report.py [results/phase5]
"""

import glob
import json
import sys
from pathlib import Path

TASKS = ("stationary", "switching", "dropout")
ARMS = ("mamba", "mamba_ablated", "gru")
LABEL = {"mamba": "selective Mamba", "mamba_ablated": "Mamba ablation", "gru": "GRU"}
COLOR = {"mamba": "#2a78d6", "mamba_ablated": "#eb6834", "gru": "#1baf7a"}


def load(d: Path) -> dict:
    out = {}
    for p in sorted(glob.glob(str(d / "*_L*_s*.json"))):
        r = json.load(open(p))
        if r.get("complete", True) and "final_val_mse" in r:
            out.setdefault((r["task"], r["arch"], r["n_layers"]), {})[r["seed"]] = r
    return out


def gate_52(cells: dict) -> dict:
    sel_a = cells.get(("stationary", "mamba", 1), {})
    abl_a = cells.get(("stationary", "mamba_ablated", 1), {})
    sel_b = cells.get(("switching", "mamba", 1), {})
    abl_b = cells.get(("switching", "mamba_ablated", 1), {})
    if min(len(sel_a), len(abl_a), len(sel_b), len(abl_b)) < 3:
        return {"verdict": "INCOMPLETE"}
    ma = sum(r["final_val_mse"] for r in sel_a.values()) / 3
    mb = sum(r["final_val_mse"] for r in abl_a.values()) / 3
    a_ok = abs(mb - ma) / ma <= 0.10
    ratios = {s: abl_b[s]["final_val_mse"] / sel_b[s]["final_val_mse"] for s in sel_b if s in abl_b}
    b_ok = len(ratios) == 3 and all(v >= 1.15 for v in ratios.values())
    return {"verdict": "PASS" if a_ok and b_ok else "FAIL",
            "a": {"selective": ma, "ablated": mb, "rel_diff": (mb - ma) / ma, "ok": a_ok},
            "b": {"ablated_over_selective": ratios, "ok": b_ok}}


def gate_53(gain: dict) -> dict:
    rs = {k: v["pearson_r"] for k, v in gain.get("models", {}).items()
          if v["arch"] == "mamba" and v["n_layers"] == 1}
    if len(rs) < 3:
        return {"verdict": "INCOMPLETE", "r": rs}
    return {"verdict": "PASS" if all(v > 0.7 for v in rs.values()) else "FAIL", "r": rs}


def plots(d: Path, gain: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    K = np.array(gain["kalman_gain_sample"]).ravel()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, arm in zip(axes, ARMS):
        key = f"switching_{arm}_L1_s0"
        m = gain["models"].get(key)
        if m:
            ax.scatter(K, np.array(m["gain_sample"]).ravel(), s=4, alpha=0.4, color=COLOR[arm])
            ax.set_title(f"{LABEL[arm]} (r = {m['pearson_r']:.2f})", fontsize=9)
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
        ax.set_xlabel("Kalman gain K_t")
        ax.grid(color="#e6e6e6", lw=0.6)
    axes[0].set_ylabel("effective gain d estimate_t / d y_t")
    fig.suptitle("Switching noise, 1 layer, seed 0, held-out sequences", fontsize=10)
    fig.tight_layout(); fig.savefig(d / "gain_scatter.png", dpi=150); plt.close(fig)

    sel = [v for k, v in gain["models"].items() if v["arch"] == "mamba" and v["n_layers"] == 1]
    if sel:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        for i, v in enumerate(sel):
            lo = np.array(v["delta"][0]["low_per_channel"]); hi = np.array(v["delta"][0]["high_per_channel"])
            ax.scatter(lo, hi, s=8, alpha=0.6, color=COLOR["mamba"], label="channels" if i == 0 else None)
        lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color="#999", lw=1, ls="--")
        ax.set_xlabel("mean Delta, low-noise regime"); ax.set_ylabel("mean Delta, high-noise regime")
        ax.set_title("Selective Mamba, 1 layer, 3 seeds: Delta per channel", fontsize=9)
        ax.grid(color="#e6e6e6", lw=0.6)
        fig.tight_layout(); fig.savefig(d / "delta_regime.png", dpi=150); plt.close(fig)

    ex = gain["example"]
    est = ex["estimates"].get("switching_mamba_L1_s0")
    if est:
        t = np.arange(len(ex["y"]))
        fig, ax = plt.subplots(figsize=(10, 3.6))
        f = np.array(ex["flag"])
        ax.fill_between(t, min(ex["y"]), max(ex["y"]), where=f == 1, color="#eee", step="mid",
                        label="high-noise regime")
        ax.scatter(t, ex["y"], s=4, color="#999", label="observation")
        ax.plot(t, ex["x"], color="black", lw=1.5, ls="--", label="latent")
        ax.plot(t, ex["kalman"], color="#2a78d6", lw=1.5, label="Kalman")
        ax.plot(t, est, color="#eb6834", lw=1.5, label="selective Mamba")
        ax.set_xlabel("step"); ax.legend(fontsize=7, frameon=False, ncol=5, loc="upper left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); fig.savefig(d / "example_trace.png", dpi=150); plt.close(fig)


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase5")
    cells = load(d)
    lines = ["## 5.2 MSE against exact baselines", "",
             "| task | model | layers | MSE mean [min, max] | / Kalman | / constant gain |",
             "|---|---|---|---|---|---|"]
    for task in TASKS:
        for L in (1, 2):
            for arm in ARMS:
                rs = list(cells.get((task, arm, L), {}).values())
                if not rs:
                    continue
                m = [r["final_val_mse"] for r in rs]
                mu = sum(m) / len(m)
                lines.append(f"| {task} | {LABEL[arm]} | {L} | {mu:.4f} [{min(m):.4f}, {max(m):.4f}] "
                             f"| {mu / rs[0]['kalman_mse']:.2f} | {mu / rs[0]['const_gain_mse']:.2f} |")
    g52 = gate_52(cells)
    lines.append(f"\nGate 5.2 ((a) stationary ablation within 10%; (b) switching ablation >= 15% "
                 f"worse on 3/3 seeds; 1 layer): **{g52['verdict']}**")
    if "a" in g52:
        lines.append(f"- (a) relative difference {g52['a']['rel_diff']:+.1%} ({'ok' if g52['a']['ok'] else 'MISS'})")
        lines.append("- (b) ablated / selective by seed: " + ", ".join(
            f"s{s} {v:.2f}" for s, v in sorted(g52["b"]["ablated_over_selective"].items()))
            + f" ({'ok' if g52['b']['ok'] else 'MISS'})")
    summary = {"5.2": g52}
    gp = d / "gain.json"
    if gp.exists():
        gain = json.load(open(gp))
        g53 = gate_53(gain)
        summary["5.3"] = g53
        lines += ["", "## 5.3 effective gain vs Kalman gain (switching, held-out)", "",
                  f"Kalman gain: low-noise {gain['kalman_gain_low']:.3f}, high-noise {gain['kalman_gain_high']:.3f}.", "",
                  "| model | Pearson r with K_t | mean gain, low noise | mean gain, high noise |",
                  "|---|---|---|---|"]
        for k, v in sorted(gain["models"].items()):
            lines.append(f"| {k} | {v['pearson_r']:.3f} | {v['gain_low']:.3f} | {v['gain_high']:.3f} |")
        lines.append(f"\nGate 5.3 (selective Mamba, 1 layer, r > 0.7 on 3/3 seeds): **{g53['verdict']}**")
        sel = [v for v in gain["models"].values() if v["arch"] == "mamba" and v["n_layers"] == 1]
        for v in sel:
            dl = v["delta"][0]
            lines.append(f"- Delta (layer 1): low-noise mean {dl['low_mean']:.4f} (channel std "
                         f"{dl['low_channel_std']:.4f}), high-noise {dl['high_mean']:.4f} "
                         f"({dl['high_channel_std']:.4f}); channels higher in low noise "
                         f"{dl['frac_channels_higher_in_low_noise']:.0%}")
        plots(d, gain)
    json.dump(summary, open(d / "summary.json", "w"), indent=2)
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
