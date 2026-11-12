"""
plan2.md 1.2 report for one setting directory, from saved JSON only: pilots
(mamba_ssm selective, Transformer) with steps-to-0.95 and the CAP; the ablated arm and
the gate once those runs exist. Writes report.md and summary.json in the directory.

Usage:
    python scripts/phase1_2_report.py results/phase1_2/seq256_nd4
    # gate re-evaluated with ablated runs from another directory (report written there)
    python scripts/phase1_2_report.py results/phase1_2/seq256_nd4 \
        results/phase1_2/seq256_nd4_sched20k
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from phase1_1_report import cap_from_pilot, steps_to

TARGET, GAP = 0.95, 0.5


def load(d: Path, prefix: str) -> list:
    return sorted((json.load(open(p)) for p in glob.glob(str(d / f"{prefix}_s*.json"))),
                  key=lambda r: r["seed"])


def setting_gate(selective: list, ablated: list) -> dict:
    sel = [r for r in selective if r.get("complete", True)]
    abl = [r for r in ablated if r.get("complete", True)]
    if len(sel) < 3 or len(abl) < 3:
        return {"verdict": "INCOMPLETE"}
    fs = [r["history"][-1]["val_acc"] for r in sel]
    fa = [r["history"][-1]["val_acc"] for r in abl]
    diff = sum(fs) / 3 - sum(fa) / 3
    ok = all(a >= TARGET for a in fs) and diff >= GAP
    return {"verdict": "PASS" if ok else "FAIL", "selective": fs, "ablated": fa,
            "mean_diff": diff}


def main():
    d = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else d
    sel, tf = load(d, "mamba_ssm_selective"), load(d, "transformer")
    abl = load(out, "mamba_ssm_ablated")
    lines = [f"## {d.name}" + (f" (ablated arm from {out.name})" if out != d else ""), "",
             "| arm | seed | schedule | steps-to-0.95 | steps run | early stop | final val_acc | wall s |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, rs in (("mamba_ssm selective", sel), ("transformer", tf), ("mamba_ssm ablated", abl)):
        for r in rs:
            s = steps_to(r)
            lines.append(f"| {arm} | {r['seed']} | {r.get('schedule_steps', r['max_steps'])} "
                         f"| {s if s is not None else 'not reached'} "
                         f"| {r['steps_run']} | {r['stopped_early']} "
                         f"| {r['history'][-1]['val_acc']:.4f}"
                         f"{'' if r.get('complete', True) else ' (partial)'} "
                         f"| {r['wall_clock_s']:.0f} |")
    sel_steps = [steps_to(r) for r in sel if r.get("complete", True)]
    cap = cap_from_pilot(sel_steps) if len(sel_steps) == 3 else None
    lines.append(f"\nCAP = {cap if cap is not None else 'undefined (pilot incomplete or a seed never reached 0.95)'}")
    g = setting_gate(sel, abl)
    lines.append(f"\nGate: **{g['verdict']}**")
    json.dump({"cap": cap, "selective_steps_to_095": sel_steps,
               "transformer_steps_to_095": [steps_to(r) for r in tf], "gate": g},
              open(out / "summary.json", "w"), indent=2)
    md = "\n".join(lines)
    (out / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
