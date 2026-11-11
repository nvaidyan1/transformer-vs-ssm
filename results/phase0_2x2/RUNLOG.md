# Phase 0.2 run log

## Run 1, 2026-11-11, commit 207ed3c, Kaggle T4, ~53 GPU-min

All 12 official runs and custom seed 0 (4 runs) finished. Custom seeds 1 (killed at the
50-minute deadline, ~21 min in) and 2 (never started) did not. Runs used `--quiet` and
the trainer wrote JSON only at the end, so the killed runs left no data.
Gate: INCOMPLETE. Stopped under rule 3. Manifest: `manifest.json`.

## Run 2, approved by the author as a second 60 GPU-min budget

Only the 8 missing custom runs (seeds 1, 2; all four cells). No change to the task,
the eval, the model, the optimizer or any training argument. Changes, logged per rule 5:
- `src/train_task.py` checkpoints its JSON record at every eval (`complete: false`)
  and writes the final record with `complete: true`. Output only, no effect on training.
- `scripts/phase0_2x2_report.py` excludes `complete: false` records from the table and
  gate and draws them dashed.
- Runner: `--quiet` dropped; deadline 54 min. Manifest: `manifest_run2.json`.

### Run 2 outcome, commit 651d51f, ~42 GPU-min

All 8 runs finished (39.6 min of training, 98 tests passed on Kaggle). Total for 0.2 so
far: ~95 GPU-min over two approved budgets.

Gate: **FAIL** in one cell, `emb_init_std=0.02, decay_1d=True`. Custom range
[0.086, 0.353] vs official [0.035, 0.079] at step 2000: no overlap (by 0.007), mean
difference +0.130 (tolerance 0.1). Custom is ABOVE official in that cell. The other
three cells pass on overlap. Per plan2.md and rule 4: stopped, no tuning. The
prescribed weight-copy follow-up has not been run.
