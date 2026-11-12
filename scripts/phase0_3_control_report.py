"""
plan2.md 0.3 Transformer control report, from saved JSON only.
Gate (results/phase0_3_lr/PREDICTION.md, Part B): mean val_acc at step 4000 over 3
seeds > 0.95. Writes summary.json and table.md next to the records.

Usage:
    python scripts/phase0_3_control_report.py [results/phase0_3_lr/control]
"""

import json
import sys
from pathlib import Path

FINAL_STEP = 4000
GATE = 0.95
N_SEEDS = 3


def summarize(records: list) -> dict:
    done = [r for r in records if r.get("complete", True) and r["history"]
            and r["history"][-1]["step"] == FINAL_STEP]
    accs = {r["seed"]: r["history"][-1]["val_acc"] for r in done}
    if len(accs) < N_SEEDS:
        return {"per_seed": accs, "gate": "INCOMPLETE"}
    vals = list(accs.values())
    mean = sum(vals) / len(vals)
    return {"per_seed": accs, "mean": mean, "min": min(vals), "max": max(vals),
            "lr": done[0]["lr"], "n_params": done[0]["n_params"],
            "gate": "PASS" if mean > GATE else "FAIL"}


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase0_3_lr/control")
    records = [json.load(open(p)) for p in sorted(d.glob("transformer_s*.json"))]
    s = summarize(records)
    json.dump(s, open(d / "summary.json", "w"), indent=2)
    lines = ["| seed | val_acc @4000 |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" for k, v in sorted(s["per_seed"].items())]
    if "mean" in s:
        lines.append(f"\nlr {s['lr']:g}, {s['n_params']:,} params. Mean {s['mean']:.4f} "
                     f"[min {s['min']:.4f}, max {s['max']:.4f}]. Gate (mean > {GATE}): "
                     f"**{s['gate']}**")
    else:
        lines.append(f"\nGate: **{s['gate']}**")
    md = "\n".join(lines)
    (d / "table.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
