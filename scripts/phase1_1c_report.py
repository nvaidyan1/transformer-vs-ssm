"""
plan2.md 1.1c report, from saved JSON only. Gate: run 2 (fixed positions, n_data=1,
mamba_ssm ablated) >= 0.9 at the final eval. Writes report.md and summary.json.

Usage:
    python scripts/phase1_1c_report.py [results/phase1_1c]
"""

import json
import sys
from pathlib import Path

GATE = 0.9
GENERATOR_NOTE = ("Non-data positions: one blank token (id 0). The generator starts from "
                  "np.full(seq_len, NOISE_ID) and overwrites only the data tokens and the "
                  "marker; there are no random noise tokens.")
RUNS = [("run1_custom_fixed_nd4", "1", "custom ablated, fixed, n_data=4 (1.1b re-run)"),
        ("run2_fixed_nd1", "2", "mamba_ssm ablated, fixed, n_data=1 (gate)"),
        ("run3_random_nd1", "3", "mamba_ssm ablated, random, n_data=1"),
        ("run4_fixed_nd2", "4", "mamba_ssm ablated, fixed, n_data=2")]


def gate(run2) -> str:
    if run2 is None or not run2.get("complete", True):
        return "INCOMPLETE"
    return "PASS" if run2["history"][-1]["val_acc"] >= GATE else "FAIL"


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1_1c")
    load = lambda n: json.load(open(d / f"{n}.json")) if (d / f"{n}.json").exists() else None
    lines = [f"0. {GENERATOR_NOTE}", "",
             "| run | what | steps run | early stop | final val_acc | per-slot accuracy |",
             "|---|---|---|---|---|---|"]
    recs = {}
    for name, num, what in RUNS:
        r = load(name)
        recs[name] = r
        if r is None:
            lines.append(f"| {num} | {what} | not run | | | |")
            continue
        slots = ", ".join(f"{a:.3f}" for a in r.get("slot_acc", []))
        lines.append(f"| {num} | {what} | {r['steps_run']} | {r['stopped_early']} "
                     f"| {r['history'][-1]['val_acc']:.4f}"
                     f"{'' if r.get('complete', True) else ' (partial)'} | {slots} |")
    g = gate(recs["run2_fixed_nd1"])
    lines.append(f"\nGate (run 2 >= {GATE}): **{g}**. Chance is 1/62 = {1/62:.4f}.")
    json.dump({"gate": g, "generator": GENERATOR_NOTE,
               "slot_acc": {k: (v or {}).get("slot_acc") for k, v in recs.items()}},
              open(d / "summary.json", "w"), indent=2)
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
