# Phase 1.1 run log

## Stage A, Stage B: 2026-11-11, commit 6f11ebd, Kaggle T4, ~10 GPU-min

As planned in PREDICTION.md. Stage A ~7 min (cap 10): the 3-process concurrency point was
skipped by the stage deadline. Stage B ~3 min (cap 20). CAP = 3000. See report.md.

## Change before Stage C (author's decision, logged per rule 5)

`configs/recall/mamba_small.yaml` (custom model), set by the author at the Stage C
approval, before any Stage C run:
- lr 3e-4 -> 1e-3; emb_init_std 0.02 -> 1.0 (out_proj depth scale on and decay_1d off,
  both unchanged). Same recipe as `mamba_ssm_small.yaml`. No CLI overrides.
- Execution: scan sequential -> parallel, activation checkpointing on -> off (Stage A:
  fastest, parity 5.96e-07 logits / 1.28e-06 relative grads). Memory/time only.
Reason: the CAP comes from the mamba_ssm pilot at lr 1e-3, emb_init_std 1.0; the old
custom recipe reached only 0.09-0.35 by step 2000 in 0.2, so a FAIL under it would have
measured the recipe, not selectivity. No Stage C arm had been run under the old config;
both Stage C arms run under the new one. The 0.2 custom results (old recipe) stand as
recorded and are not rerun: they answer a different question (0.2).

Stage C budget: 70 GPU-min total (author). One process at a time.

## Stage C: 2026-11-11, commit c524115, Kaggle T4, ~63 GPU-min (budget 70)

All 8 runs complete, one at a time (~7.8 min each). Ablated mean at lr 1e-3 was 0.019,
below 0.5, so the two extra seed-0 ablated runs (lr 3e-4, 3e-3) ran as specified.
Gate: **PASS**. Selective 0.996 / 1.000 / 0.998 (steps-to-0.95: 1500 / 1500 / 2000);
best ablated lr 3e-3 at 0.029; difference +0.969. See report.md, curves_stage_c.png.
Caveat for comparisons with the reference arm: the mamba_ssm pilot ran with a
20,000-step cosine schedule (Stage B), the custom arms with a 3000-step one, so their
lr at a given step differs.
