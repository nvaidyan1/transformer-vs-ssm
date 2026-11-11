"""Gate logic for plan2.md 0.2 (scripts/phase0_2x2_report.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase0_2x2_report import acc_at, cell_gate, stats, summarize, to_markdown


def rec(arch, emb, d1d, seed, acc1000, acc2000):
    return {"arch": arch, "emb_init_std": emb, "decay_1d": d1d, "seed": seed,
            "chance_acc": 1 / 31,
            "history": [{"step": 1000, "val_acc": acc1000},
                        {"step": 2000, "val_acc": acc2000}]}


def s(*vals):
    return stats(list(vals))


def test_overlap_passes_even_with_large_mean_gap():
    g = cell_gate(s(0.1, 0.2, 0.9), s(0.85, 0.95, 0.99))
    assert g["overlap"] and g["verdict"] == "PASS"


def test_close_means_pass_without_overlap():
    g = cell_gate(s(0.50, 0.51, 0.52), s(0.58, 0.59, 0.60))
    assert not g["overlap"] and abs(g["mean_diff"]) <= 0.1 and g["verdict"] == "PASS"


def test_disjoint_and_far_fails():
    g = cell_gate(s(0.10, 0.12, 0.13), s(0.45, 0.46, 0.50))
    assert not g["overlap"] and g["verdict"] == "FAIL"


def test_fewer_than_three_seeds_is_incomplete():
    assert cell_gate(s(0.5, 0.5), s(0.5, 0.5, 0.5))["verdict"] == "INCOMPLETE"


def test_acc_at_missing_step_is_none():
    assert acc_at(rec("mamba", 1.0, False, 0, 0.3, 0.4), 1500) is None


def test_summarize_four_cells_overall_verdict():
    recs = []
    for emb in (0.02, 1.0):
        for d1d in (False, True):
            for seed in range(3):
                recs.append(rec("mamba_ssm", emb, d1d, seed, 0.3, 0.5 + 0.01 * seed))
                # one failing cell: custom far below official
                low = emb == 1.0 and d1d
                recs.append(rec("mamba", emb, d1d, seed, 0.1,
                                (0.1 if low else 0.5) + 0.01 * seed))
    out = summarize(recs)
    assert len(out["cells"]) == 4
    verdicts = {(c["emb_init_std"], c["decay_1d"]): c["gate"]["verdict"]
                for c in out["cells"]}
    assert verdicts[(1.0, True)] == "FAIL"
    assert sum(v == "PASS" for v in verdicts.values()) == 3
    assert out["gate"] == "FAIL"
    assert "Overall gate: **FAIL**" in to_markdown(out)


def test_summarize_missing_cell_is_incomplete():
    recs = [rec(a, 0.02, False, sd, 0.3, 0.5) for a in ("mamba", "mamba_ssm")
            for sd in range(3)]
    assert summarize(recs)["gate"] == "INCOMPLETE"


def test_partial_runs_excluded_from_gate():
    recs = []
    for emb in (0.02, 1.0):
        for d1d in (False, True):
            for seed in range(3):
                recs.append(rec("mamba_ssm", emb, d1d, seed, 0.3, 0.5))
                recs.append(rec("mamba", emb, d1d, seed, 0.3, 0.5))
    killed = rec("mamba", 0.02, False, 3, 0.01, 0.01)
    killed.update(complete=False, steps_run=1000)
    out = summarize(recs + [killed])
    assert out["gate"] == "PASS"
    cell = next(c for c in out["cells"] if c["emb_init_std"] == 0.02 and not c["decay_1d"])
    assert cell["mamba"]["seeds"] == [0, 1, 2]
    assert len(out["partial_runs_excluded"]) == 1
