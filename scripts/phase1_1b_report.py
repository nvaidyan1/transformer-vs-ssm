"""
plan2.md 1.1b report, from saved JSON only: the fixed-position positive control (gate:
ablated >= 0.9) and the diagnostic for the 1.1 ablated run (training loss vs ln(31),
histogram of predicted tokens at answer positions). Writes report.md, summary.json and
answer_hist.png next to the records.

Usage:
    python scripts/phase1_1b_report.py [results/phase1_1b]
"""

import json
import math
import sys
from pathlib import Path

GATE = 0.9
ORIGINAL_1_1_ACC = 0.0205078125      # results/phase1_1/stage_c/custom_ablated_s0.json


def steps_to(r, target=0.95):
    return next((h["step"] for h in r["history"] if h["val_acc"] >= target), None)


def gate(fixed_ablated: dict) -> str:
    if fixed_ablated is None or not fixed_ablated.get("complete", True):
        return "INCOMPLETE"
    return "PASS" if fixed_ablated["history"][-1]["val_acc"] >= GATE else "FAIL"


def hist_stats(h: dict) -> dict:
    pred = {int(k): v for k, v in h["pred"].items()}
    n = sum(pred.values())
    top = sorted(pred.items(), key=lambda kv: -kv[1])[:5]
    probs = [v / n for v in pred.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return {"n_answers": n, "distinct_predicted": len(pred),
            "distinct_targets": len(h["target"]),
            "non_data_predictions": sum(v for k, v in pred.items() if k < 2),
            "top5": top, "pred_entropy_nats": entropy}


def plot_hist(h: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pred = {int(k): v for k, v in h["pred"].items()}
    n = sum(pred.values())
    ids = list(range(64))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(ids, [pred.get(i, 0) for i in ids], width=0.8, color="#2a78d6")
    ax.axhline(n / 62, color="#999", lw=1, ls="--")
    ax.text(63.6, n / 62, "uniform over\\n62 data ids", fontsize=7, color="#777", va="center")
    ax.set_xlabel("predicted token id at answer positions (data ids are 2-63)")
    ax.set_ylabel("count")
    ax.set_title(f"1.1 ablated model (re-run, seed 0): predictions on {n} validation answers",
                 fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/phase1_1b")
    load = lambda n: json.load(open(d / f"{n}.json")) if (d / f"{n}.json").exists() else None
    fa, fs, rr = load("fixed_ablated_s0"), load("fixed_selective_s0"), load("rerun_custom_ablated_s0")
    lines = ["## Part 1: fixed-position Selective Copying (seed 0, CAP 3000)", "",
             "| arm | steps-to-0.95 | steps run | early stop | final val_acc |", "|---|---|---|---|---|"]
    for name, r in (("ablated", fa), ("selective", fs)):
        if r:
            s = steps_to(r)
            lines.append(f"| {name} | {s if s is not None else 'not reached'} | {r['steps_run']} "
                         f"| {r['stopped_early']} | {r['history'][-1]['val_acc']:.4f}"
                         f"{'' if r.get('complete', True) else ' (partial)'} |")
    g = gate(fa)
    lines.append(f"\nGate (ablated >= {GATE}): **{g}**")
    summary = {"gate": g}
    if rr:
        st = hist_stats(rr["answer_hist"])
        tl = rr["history"][-1]["train_loss"]
        summary.update(rerun_final_val_acc=rr["history"][-1]["val_acc"], rerun_train_loss=tl,
                       hist=st)
        lines += ["", "## Part 2: the 1.1 ablated run (re-run with the identical command)", "",
                  f"- Final val_acc {rr['history'][-1]['val_acc']:.4f} (original 1.1 run: "
                  f"{ORIGINAL_1_1_ACC:.4f}); 1/62 = {1/62:.4f}, the trainer's chance_acc = 1/31 = {1/31:.4f}.",
                  f"- Final training loss {tl:.3f} (last logged batch) vs ln(31) = {math.log(31):.3f} "
                  f"and ln(62) = {math.log(62):.3f}.",
                  f"- Predictions: {st['distinct_predicted']} distinct tokens over "
                  f"{st['n_answers']} answers ({st['distinct_targets']} distinct targets); "
                  f"{st['non_data_predictions']} non-data predictions; entropy "
                  f"{st['pred_entropy_nats']:.3f} nats (uniform over 62 = {math.log(62):.3f}); "
                  f"top 5 (id, count): {st['top5']}."]
        plot_hist(rr["answer_hist"], d / "answer_hist.png")
    json.dump(summary, open(d / "summary.json", "w"), indent=2)
    md = "\n".join(lines)
    (d / "report.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
