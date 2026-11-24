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

## 2026-11-12: protocol change, one lr-schedule length for every arm (author's decision)

From now on every arm in every setting uses a 20,000-step lr schedule (200-step warmup,
then cosine to 10% of peak over 20,000 steps). CAP and early stopping only decide when a
run stops; they never change the schedule. Implemented as `src/train_task.py
--schedule_steps` (recorded in each result as `schedule_steps`); its default is still the
stop step, so commands written before this date reproduce exactly.

Schedule lengths used before the change (from the saved records, `max_steps`):

| Rung | Arm | Schedule length |
|---|---|---|
| 1.1 Stage B | mamba_ssm selective (reference) | 20,000 |
| 1.1 Stage C | custom selective, custom ablated (all lrs) | 3,000 |
| 1.1d | selective and ablated, every n_data | 4,000 |
| 1.2 seq_len 256 | mamba_ssm selective pilot, Transformer pilot | 20,000 |
| 1.2 seq_len 256 | mamba_ssm ablated | 4,000 |

Also 3,000 for both arms in 1.1b, 1.1c (runs 1-4) and 1.2a; 20,000 for 1.1c's report-only
run 2 re-run. Within 1.1 Stage C, 1.1b, 1.1c, 1.1d and 1.2a the compared arms shared a
schedule; in 1.2 seq_len 256 they did not (ablated shorter), and 1.1 Stage C's reference
arm differed from its custom arms. Follow-ups (b) and (d) rerun the affected ablated cells.

## 2026-11-19: cruise decisions D1-D6 (author) and the CAP-extension rule

Recorded in plan2.md and CLAIM_MAP.md. D1: Phase 2 uses the small matched configs (~450k).
D2: Phase 3 reduced (both TCNs x 6 gaps x 3 seeds; Transformer and mamba_ssm at 2 gaps).
D4: 1.1d remaining cells rerun on the 20k schedule. D5: parameter matching stays on totals
(positional table counted); the Transformer's non-positional count (399,048, -11.1%) is
disclosed wherever matching is claimed. D6: standing approval per phase within the
CLAIM_MAP budgets; no experiment outside a CLAIM_MAP row.

D3, CAP extension rule (fixed before any Phase 3/4 run): every cell runs on the 20,000-step
schedule and stops at CAP 4000 (or earlier by the usual early stop). A cell is extended to
the full 20,000 steps only if, at CAP, it is
- **ambiguous**: final val_acc in [0.2, 0.9], or
- **unconverged**: val_acc rose by more than 0.05 over its last 1000 steps (4 evals).
A cell below 0.2 and flat, or above 0.9, is final at CAP. Extensions run once; no further
extension.

## 2026-11-19: Phase 2.1 failed gate, claims restricted (author decision)

2.1 failed on three timing slopes (SDPA 1.63 vs 2; TCN 0.60 and Mamba 0.50 vs 1). Decision:
the original results are preserved and reported as failed. Posts may use the memory scaling
(all four memory slopes as predicted) and the naive-attention timing slope (1.81, reported,
not gated); no claim rests on the SDPA, TCN or Mamba timing slopes. A larger-batch timing
rerun would be a separately logged, predeclared follow-up and would not replace these
results. Not scheduled.

## 2026-11-19: Phase 1 closed (author decision)

1.2 closed at seq_len 64 and 256 (gates PASS). seq_len 1024 is a partial pilot (mamba_ssm
selective CAP 9000 found; Transformer reference cut at step 6500). The seq_len 1024 ablated
arm and n_data 16 are follow-ups, not scheduled.

## 2026-11-23: gate completion for Phases 3 and 4; 5.3 post hoc (author approval)

Approved: P3a (seed 2 for the four tcn_short gate cells) and P4a (D3 extensions for the six
4-pair gate cells), one kernel. Not approved yet: P3b, P3c, P4b. Phase 5.3 is run after the
failed 5.2 gate as a **post-hoc descriptive analysis**, labelled as such; it decides nothing.
Phase 3 exceeded its ~3 GPU-h budget (3.5 h) and Phase 4 reached ~2.1 of 2-3 GPU-h before
this; the overrun was approved with these items.
