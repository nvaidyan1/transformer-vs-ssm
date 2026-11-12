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


def gate_c(selective: list, ablated: list) -> dict:
    fs = [r["history"][-1]["val_acc"] for r in selective]
    fa = [r["history"][-1]["val_acc"] for r in ablated]
    if len(fs) < 3 or len(fa) < 3:
        return {"verdict": "INCOMPLETE"}
    diff = sum(fs) / len(fs) - sum(fa) / len(fa)
    ok = all(a >= TARGET for a in fs) and diff >= GAP
    return {"verdict": "PASS" if ok else "FAIL", "selective": fs, "ablated": fa,
            "mean_diff": diff}


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

    sel = [json.load(open(p)) for p in glob.glob(str(d / "stage_c/custom_selective_s*.json"))]
    abl = [json.load(open(p)) for p in glob.glob(str(d / "stage_c/custom_ablated_s*.json"))]
    if sel or abl:
        g = gate_c([r for r in sel if r.get("complete", True)],
                   [r for r in abl if r.get("complete", True)])
        summary["gate"] = g
        lines += ["", f"## Stage C gate: **{g['verdict']}**"]
    json.dump(summary, open(d / "summary.json", "w"), indent=2)
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
