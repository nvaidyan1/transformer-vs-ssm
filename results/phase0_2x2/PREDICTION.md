# Phase 0.2 prediction: does init scale and bias decay explain custom vs official?

Written and committed before any 0.2 run (plan2.md, rule 1).

## Setup

- Task: `selective_copy`, `n_data=4`, `seq_len=64`, `vocab_size=64` (chance 1/31 = 0.032).
- 2000 steps, batch 32, lr 3e-4, 200-step warmup then cosine to 10%, AdamW (0.9, 0.95),
  wd 0.1 (grouped), grad clip 1.0, eval every 250 steps on a fixed 8-batch validation set.
- Seeds 0, 1, 2. The seed sets the init and the training batch stream; at a given seed the
  custom and official models see identical batches.
- Model: `configs/recall/mamba_small.yaml` (4 layers, d_model 124, d_state 16, d_conv 4,
  expand 2). Custom = arch `mamba` (sequential scan, gradient checkpointing on, the repo
  defaults). Official = arch `mamba_ssm` (mamba-ssm 2.3.2.post1, use_fast_path=True).
  Trainable parameters 448,632 in both (custom has 1,120 more, ablation-only and unused).
- Arms, for each arch: `emb_init_std` in {0.02, 1.0} x `decay_1d` in {False, True}.
- Held fixed in every cell: `out_proj_depth_scale=True`. The 2x2 does not vary it. Note the
  original oracle run (official 0.461 vs custom 0.128) used `emb_init_std=1.0`,
  `out_proj_depth_scale=False`, `decay_1d=True` for the official model, so no cell here
  reproduces it exactly.
- Code: commit containing this file; runner `scripts/kaggle/phase0_2x2/`.

## Prediction (from plan2.md 0.2)

1. Within a cell, custom and official overlap across seeds.
2. The `emb_init_std=1.0` cells learn faster than the `0.02` cells.
3. `decay_1d` has a small effect.

My own expectation, recorded for honesty: (1) is uncertain. recap.md section 6 shows a small,
one-signed, compounding divergence between the two implementations even from identical
weights, so a gap at 2000 steps in some cells would not be surprising.

## Gate

In every one of the 4 cells: the custom seed range [min, max] of `val_acc` at step 2000
overlaps the official seed range, OR |custom mean - official mean| <= 0.1.

## If the gate fails

In a cell that fails, do not tune. Starting from weight-copied identical models with the
identical optimizer, train both for 2000 steps on the same batches and report the two
accuracy curves. Nothing further.

## Budget

60 GPU-minutes for 0.2. Estimated from the 0.1 smoke run: official ~0.03 s/step
(12 runs, a few minutes), custom ~0.32 s/step (~11 min/run, ~128 min serial). Plan: run 4
jobs concurrently on the T4 with a hard training deadline, official runs first, custom runs
ordered seed-major so a deadline cut leaves every cell with the same number of seeds. If
the deadline cuts runs, stop and report; do not extend.
