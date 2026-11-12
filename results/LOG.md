# Results log

Changes to code, evals or reference numbers that affect how saved results are read.
Raw result JSON is never edited; prediction files are never edited after their run.

## 2026-11-12: chance accuracy for selective_copy corrected (author's decision)

`src/train_task.py` recorded `chance_acc = 1 / floor((V - 1) / 2)` for every task. That
is MQAR's answer alphabet (31 values at V = 64). Selective Copying draws data from ids
[2, V), 62 symbols at V = 64, so its chance accuracy is 1/62 = 0.016, not 1/31 = 0.032.
Fix: `src.tasks.chance_accuracy(task, vocab_size)`, used by the trainer from now on and
by every report script via `record_chance(record)`, which ignores the stored value.
Figures regenerated from the unchanged JSON: phase0_2x2, phase0_3_lr,
phase0_weight_copy, phase1_1. Affects only the chance reference line and the stored
`chance_acc` field: no metric, gate, or selection rule ever used it. Prediction files
that say "chance 0.032" are left as written. Found in 1.1b (results/phase1_1b).

## 2026-11-12: 1.1c gate, post-hoc amendment (author's decision)

The 1.1c gate as written (run 2, fixed positions, n_data = 1, >= 0.9) FAILED (0.352 at
CAP 3000). Amended after seeing the results, by the author. Reason: the gate's purpose was
"the ablated model can learn to carry one item"; run 3 (random positions, n_data = 1)
solved it (1.000), which satisfies that purpose. The ablation is cleared for use, with
claims worded as "input-dependent dynamics", not "selection", and with the per-slot result
and 1.1c runs 2-4 reported beside it. Run 2 is re-run at CAP 20,000 as a report-only
item that gates nothing. This is a post-hoc change and is labelled as such wherever the
ablation result is used.
