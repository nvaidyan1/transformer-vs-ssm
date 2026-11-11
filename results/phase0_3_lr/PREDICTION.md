# Phase 0.3 prediction: learning-rate check and Transformer control

Written and committed before any 0.3 run (rule 1). Added to plan2.md 0.3 by the author
on 2026-11-11.

## Setup common to both parts

- Task: `selective_copy`, `n_data=4`, `seq_len=64`, `vocab_size=64` (chance 0.032).
- Configs after re-matching (all within 0.8% of 448,632 trainable parameters):
  `transformer_small.yaml` (d_model 104, 452,296), `tcn_small.yaml` (d_model 98,
  450,212, receptive field 769), `mamba_ssm_small.yaml` (448,632; locked recipe
  emb_init_std 1.0, out_proj_depth_scale true, decay_1d false).
- 4000 steps, batch 32, 200-step warmup then cosine to 10% of peak, AdamW (0.9, 0.95),
  wd 0.1 grouped, grad clip 1.0, eval every 250 steps on the fixed 8-batch validation set.

## Part A: learning-rate check

lr in {3e-4, 1e-3, 3e-3}, seed 0 only (one-off, by the author's addition; this overrides
rule 2 for the lr check), for Transformer, TCN and mamba_ssm. 9 runs.

**Prediction.**
1. Higher lr helps Mamba: mamba_ssm's val_acc at step 4000 is higher at 1e-3 or 3e-3 than
   at 3e-4.
2. The Transformer is already fine at 3e-4 (val_acc at step 4000 > 0.95).
3. TCN: no prediction from the author. My expectation: lr matters less than the fixed
   routing; it may stay well below the Transformer at every lr.

**Instability rule (fixed now, applied mechanically).** A run is unstable if any of:
- any train_loss or val_loss in the eval history is NaN or infinite;
- val_acc at step 4000 is more than 0.05 below that run's maximum val_acc over all evals;
- val_loss at step 4000 is higher than val_loss at the first eval (step 250).

**Selection rule.** Per architecture, among stable runs, take the highest val_acc at step
4000. If several stable runs are within 0.005 of the best, take the lowest of those lr.
If every run of an architecture is unstable, no lr is picked for it: stop and report.
The picked lr is written into that architecture's config and not changed in later phases.

**Budget.** 30 GPU-min. The Kaggle runner enforces a 24-minute training deadline after
~4 min of setup; runs cut by it keep their partial curves and count as not selectable.

## Part B: Transformer control

Transformer at its picked lr, seeds 0, 1, 2, 4000 steps, from the committed config.

**Prediction.** It solves the task on every seed.

**Gate (0.3).** Mean val_acc at step 4000 over the 3 seeds > 0.95 (min also reported);
parameter counts within 5% (checked by `tests/test_recall_configs.py`); tests pass.

**Budget.** Outside the lr-check's 30 minutes; expected ~8 GPU-min including setup.
