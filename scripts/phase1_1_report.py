"""
plan2.md 1.1 (amended) report, from saved JSON only.

Stage A: per-configuration speed, parity of the fastest, concurrency throughput.
Stage B: mamba_ssm pilot steps-to-0.95 per seed, and the Stage C CAP.
Stage C projection: GPU-minutes for 6 custom runs to CAP at a given concurrency.
Stage C gate (once those runs exist): selective >= 0.95 on 3/3 AND selective - ablated
>= 0.5 (seed-mean) at the final eval.

Usage:
    python scripts/phase1_1_report.py [results/phase1_1]
"""

import glob
import json
import math
import sys
from pathlib import Path

TARGET = 0.95
CAP_FACTOR = 1.5
CAP_MAX = 20_000
GAP = 0.5


def steps_to(record: dict, target: float = TARGET):
    for h in record["history"]:
        if h["val_acc"] >= target:
            return h["step"]
    return None


def cap_from_pilot(steps: list):
    """None if any seed never reached the target (stop, per the amendment)."""
    if not steps or any(s is None for s in steps):
        return None
    return min(CAP_MAX, int(math.ceil(CAP_FACTOR * max(steps) / 1000.0)) * 1000)


def concurrency_table(stage_a: Path) -> dict:
    out = {}
    for p in glob.glob(str(stage_a / "conc_k*_p*.json")):
        r = json.load(open(p))
        k = int(Path(p).name.split("_")[1][1:])
        out.setdefault(k, []).append(r)
    table = {}
    for k, rs in sorted(out.items()):
        if len(rs) != k:
            continue
        per = [r["steps_per_s"] for r in rs]
        starts = [r["t_start"] for r in rs]
        table[k] = {"per_process_steps_per_s": min(per), "aggregate_steps_per_s": sum(per),
                    "start_spread_s": max(starts) - min(starts)}
    return table


def project_gpu_min(cap: int, n_runs: int, k: int, per_process_steps_per_s: float,
                    setup_min: float = 3.0, eval_overhead: float = 0.05) -> float:
    """Worst case (no early stop): ceil(n_runs/k) waves, each CAP steps long at the
    per-process rate measured with k processes sharing the GPU."""
    waves = math.ceil(n_runs / k)
    return setup_min + waves * cap / per_process_steps_per_s / 60.0 * (1 + eval_overhead)


def gate_c(selective: list, ablated: list, ablated_extra: dict = None) -> dict:
    """selective, ablated: the 3-seed runs at the config lr. ablated_extra:
    {lr: [records]} for the extra seed-0 ablated runs (amendment item 3). The gate
    compares the selective mean against the best ablated lr, where each lr's value is
    the mean final val_acc of its runs."""
    fs = [r["history"][-1]["val_acc"] for r in selective]
    fa = [r["history"][-1]["val_acc"] for r in ablated]
    if len(fs) < 3 or len(fa) < 3:
        return {"verdict": "INCOMPLETE"}
    by_lr = {ablated[0]["lr"]: sum(fa) / len(fa)}
    for lr, rs in (ablated_extra or {}).items():
        if rs:
            by_lr[lr] = sum(r["history"][-1]["val_acc"] for r in rs) / len(rs)
    best_lr = max(by_lr, key=by_lr.get)
    diff = sum(fs) / len(fs) - by_lr[best_lr]
    ok = all(a >= TARGET for a in fs) and diff >= GAP
    return {"verdict": "PASS" if ok else "FAIL", "selective": fs, "ablated": fa,
            "ablated_by_lr": by_lr, "best_ablated_lr": best_lr, "mean_diff": diff}


C_REF, C_SEL, C_ABL = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3


def plot_c(pilots, sel, abl, extra, path, chance):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2))

    def curve(r, color, ls, label):
        ax.plot([h["step"] for h in r["history"]], [h["val_acc"] for h in r["history"]],
                color=color, lw=2 if ls == "-" else 1.6, ls=ls, alpha=0.9, label=label)

    for i, r in enumerate(pilots):
        curve(r, C_REF, "-", "mamba_ssm selective (reference, 3 seeds)" if i == 0 else None)
    for i, r in enumerate(sel):
        curve(r, C_SEL, "-", "custom selective (3 seeds)" if i == 0 else None)
    for i, r in enumerate(abl):
        curve(r, C_ABL, "--", "custom ablated, lr 1e-3 (3 seeds)" if i == 0 else None)
    for i, r in enumerate(x for v in extra.values() for x in v):
        curve(r, C_ABL, ":", "custom ablated, lr 3e-4 / 3e-3 (seed 0)" if i == 0 else None)
    ax.axhline(0.95, color="#999", lw=1)
    ax.text(3020, 0.95, "0.95", fontsize=7, color="#777", va="center")
    ax.axhline(chance, color="#999", lw=1, ls="--")
    ax.text(3020, chance, "chance", fontsize=7, color="#777", va="center")
    ax.axvline(3000, color="#bbb", lw=1, ls=":")
    ax.set_xlim(0, 3300)
    ax.set_xlabel("step (CAP = 3000)")
    ax.set_ylabel("val_acc (answer positions)")
    ax.set_title("Selective Copying, n_data=4, seq_len=64", fontsize=10)
    ax.grid(color="#e6e6e6", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def by_best(g):
    return g["ablated_by_lr"][g["best_ablated_lr"]]


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1_1")
    lines = ["## Stage A: custom Mamba training step (seq_len 64, batch 32)", "",
             "| scan | grad_checkpoint | ms/step | steps/s | peak MiB |", "|---|---|---|---|---|"]
    cfgs = sorted((json.load(open(p)) for p in glob.glob(str(d / "stage_a/config_*.json"))),
                  key=lambda r: -r["steps_per_s"])
    for r in cfgs:
        lines.append(f"| {r['scan']} | {r['grad_checkpoint']} | {r['ms_per_step']:.1f} "
                     f"| {r['steps_per_s']:.2f} | {r['peak_mem_mib']:.0f} |")
    for p in glob.glob(str(d / "stage_a/parity_*.json")):
        par = json.load(open(p))
        q = par["parity"]
        lines.append(f"\nParity, {par['scan']} / checkpointing {par['grad_checkpoint']} vs "
                     f"sequential / on (CUDA): max |logit diff| {q['max_abs_logit_diff']:.2e}, "
                     f"max relative grad diff {q['max_rel_grad_diff']:.2e} over "
                     f"{q['n_grads']} tensors.")
    conc = concurrency_table(d / "stage_a")
    if conc:
        lines += ["", "| processes | per-process steps/s | aggregate steps/s | speedup |",
                  "|---|---|---|---|"]
        base = conc.get(1, {}).get("aggregate_steps_per_s")
        for k, c in conc.items():
            sp = f"{c['aggregate_steps_per_s']/base:.2f}x" if base else "n/a"
            lines.append(f"| {k} | {c['per_process_steps_per_s']:.2f} "
                         f"| {c['aggregate_steps_per_s']:.2f} | {sp} |")

    pilots = sorted((json.load(open(p)) for p in glob.glob(str(d / "stage_b/mamba_ssm_s*.json"))),
                    key=lambda r: r["seed"])
    steps = [steps_to(r) for r in pilots]
    lines += ["", "## Stage B: mamba_ssm pilot", "",
              "| seed | steps-to-0.95 | steps run | early stop | final val_acc |",
              "|---|---|---|---|---|"]
    for r, s in zip(pilots, steps):
        lines.append(f"| {r['seed']} | {s if s is not None else 'not reached'} "
                     f"| {r['steps_run']} | {r['stopped_early']} | {r['history'][-1]['val_acc']:.4f} |")
    cap = cap_from_pilot(steps) if len(pilots) == 3 else None
    summary = {"stage_a": cfgs, "concurrency": conc, "pilot_steps_to_095": steps, "cap": cap}
    if cap is not None:
        lines.append(f"\nCAP = ceil(1.5 x {max(steps)} / 1000) x 1000 = **{cap}**")
        if conc:
            proj = {k: project_gpu_min(cap, 6, k, c["per_process_steps_per_s"])
                    for k, c in conc.items()}
            summary["projection_gpu_min"] = proj
            lines += ["", "Stage C projection (6 runs to CAP, no early stop, +3 min setup):", ""]
            lines += [f"- {k} process(es): {v:.0f} GPU-min" for k, v in proj.items()]
    else:
        lines.append("\nCAP undefined: not every pilot seed reached 0.95. Stop (see PREDICTION.md).")

    def load_c(pattern):
        return sorted((json.load(open(p)) for p in glob.glob(str(d / "stage_c" / pattern))),
                      key=lambda r: r["seed"])
    sel, abl = load_c("custom_selective_s*.json"), load_c("custom_ablated_s*.json")
    extra = {}
    for r in load_c("custom_ablated_lr*_s0.json"):
        if r.get("complete", True):
            extra.setdefault(r["lr"], []).append(r)
    if sel or abl:
        lines += ["", "## Stage C: custom Mamba, CAP 3000 steps", "",
                  "| arm | lr | seed | steps-to-0.95 | steps run | early stop | final val_acc |",
                  "|---|---|---|---|---|---|---|"]
        for arm, rs in (("selective", sel), ("ablated", abl),
                        ("ablated", [r for v in extra.values() for r in v])):
            for r in rs:
                s = steps_to(r)
                lines.append(f"| {arm} | {r['lr']:g} | {r['seed']} "
                             f"| {s if s is not None else 'not reached'} | {r['steps_run']} "
                             f"| {r['stopped_early']} | {r['history'][-1]['val_acc']:.4f}"
                             f"{'' if r.get('complete', True) else ' (partial)'} |")
        if pilots:
            plot_c(pilots, sel, abl, extra, d / "curves_stage_c.png",
                   pilots[0].get("chance_acc", 1 / 31))
        g = gate_c([r for r in sel if r.get("complete", True)],
                   [r for r in abl if r.get("complete", True)], extra)
        summary["gate"] = g
        lines += ["", f"## Stage C gate: **{g['verdict']}**"]
        if g["verdict"] != "INCOMPLETE":
            fs = g["selective"]
            by = ", ".join(f"lr {k:g}: {v:.3f}" for k, v in sorted(g["ablated_by_lr"].items()))
            lines += ["", f"Selective final val_acc {', '.join(f'{a:.3f}' for a in fs)} "
                          f"(>= 0.95 on {sum(a >= TARGET for a in fs)}/3). Ablated final "
                          f"val_acc by lr ({by}); best lr {g['best_ablated_lr']:g}. "
                          f"Selective mean - best ablated = {g['mean_diff']:+.3f} "
                          f"(needs >= {GAP})."]
            if g["verdict"] == "PASS":
                lines += ["", "Claim, as the evidence supports it: within the step budget at "
                              "which the selective model solves the task (3000 steps), the "
                              "custom selective Mamba reaches >= 0.95 on 3/3 seeds and the "
                              "input-independent ablation does not, at any lr tried "
                              f"(best {by_best(g):.3f})."]
    json.dump(summary, open(d / "summary.json", "w"), indent=2)
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
