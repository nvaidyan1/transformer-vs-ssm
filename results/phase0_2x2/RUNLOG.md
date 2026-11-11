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
