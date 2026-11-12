"""
plan2.md 1.2a report, from saved JSON only. Gate: official selective >= 0.95 AND
official ablated < 0.2 at the final eval. Writes report.md and summary.json.

Usage:
    python scripts/phase1_2a_report.py [results/phase1_2a]
"""

import json
import sys
from pathlib import Path


def steps_to(r, target=0.95):
    return next((h["step"] for h in r["history"] if h["val_acc"] >= target), None)


def gate(sel, abl) -> str:
    if not sel or not abl or not sel.get("complete", True) or not abl.get("complete", True):
        return "INCOMPLETE"
    ok = sel["history"][-1]["val_acc"] >= 0.95 and abl["history"][-1]["val_acc"] < 0.2
    return "PASS" if ok else "FAIL"


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1_2a")
    load = lambda n: json.load(open(d / f"{n}.json")) if (d / f"{n}.json").exists() else None
    sel, abl = load("official_selective_s0"), load("official_ablated_s0")
    lines = ["| arm (mamba_ssm, seed 0) | steps-to-0.95 | steps run | early stop | final val_acc |",
             "|---|---|---|---|---|"]
    for name, r in (("selective", sel), ("ablated", abl)):
        if r:
            s = steps_to(r)
            lines.append(f"| {name} | {s if s is not None else 'not reached'} | {r['steps_run']} "
                         f"| {r['stopped_early']} | {r['history'][-1]['val_acc']:.4f} |")
    g = gate(sel, abl)
    lines.append(f"\nGate (selective >= 0.95 AND ablated < 0.2): **{g}**")
    json.dump({"gate": g}, open(d / "summary.json", "w"), indent=2)
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
