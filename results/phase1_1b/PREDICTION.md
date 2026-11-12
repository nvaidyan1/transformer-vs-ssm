# Phase 1.1b prediction: positive control for the ablation

Written and committed before any 1.1b run (rule 1). Added by the author, 2026-11-11.
Budget: 25 GPU-min.

## Part 1: fixed-position task

`selective_copy` with `fixed_positions=1`: the 4 data tokens sit at positions
0, 19, 39, 58 of the 59-token input region in every sequence (vanilla copying; a
time-invariant model can solve it with fixed taps). seq_len 64, vocab 64, n_data 4.
Custom Mamba (`configs/recall/mamba_small.yaml`, locked recipe: lr 1e-3, emb_init_std
1.0, parallel scan, no checkpointing), seed 0, CAP 3000 steps, early stop at val_acc
> 0.99 for 3 consecutive evals. Arms: ablated (`--ablate`) and selective.

**Prediction (author):** both models exceed 0.95 on the fixed-position task.
**Gate:** ablated >= 0.9 at the final eval. If it fails: stop and report; the 1.1
ablation result cannot be used in a post until the ablated model is shown to learn
something.

## Part 2: the 1.1 ablated model's below-chance accuracy

There is no saved checkpoint, so the 1.1 Stage C run `custom_ablated_s0` is re-run with
the identical command (config, seed 0, CAP 3000, early-stop flags) plus
`--dump_answer_hist`. Reported: whether it reproduces the original (val_acc 0.0205 at
step 3000), its final training loss against ln(31) = 3.434, and the histogram of its
predicted tokens at answer positions on the validation set.

**Prediction (mine, from reading the generator before the run):** the accuracy is not
below chance. Data tokens are drawn uniformly from ids [2, 64), i.e. 62 symbols, so
uniform guessing scores 1/62 = 0.016; the trainer's `chance_acc` = 1/31 uses the formula
floor((V-1)/2), which does not match this generator. Expected: final training loss near
ln(62) = 4.127, i.e. above ln(31); predicted tokens spread over most of the 62 data ids
(none of them the noise or marker ids). Only a reference number is wrong, not any metric
or gate; correcting `chance_acc` is left to the author.

## Budget

Three runs of at most 3000 steps, one at a time (~8 min each at 6.45 steps/s, Phase
1.1 Stage A), plus ~1 min setup: ~25 min. The kernel enforces a 23.5-minute training
deadline, ordered ablated-fixed (the gate) first, selective-fixed second, the re-run last.

## Outcome (appended after the run; predictions above unchanged)

Kaggle T4, commit f7bee46, ~22.5 GPU-min of 25. 118 tests passed.

Part 1, fixed positions: selective 1.000 (steps-to-0.95 1250, early stop at 2250).
Ablated 0.258 at step 3000; it rose from 0.045 to 0.255 between steps 750 and 1000
and stayed within 0.254-0.262 for the remaining 2000 steps. **Gate FAIL** (needs >= 0.9).
The author's prediction (both > 0.95) holds for selective and fails for ablated. Stopped
here per the gate rule; no investigation. The 1.1 ablation result is not usable in a post
until the ablated model is shown to learn something.

Part 2, re-run of 1.1 custom_ablated_s0: reproduced exactly (val_acc 0.0205 at step
3000, same as the original). Final training loss 4.142 (last logged batch), above
ln(31) = 3.434 and close to ln(62) = 4.127. Predicted tokens: 29 distinct ids over 1024
answers, all data ids, but concentrated: id 42 in 732 of 1024 (71%), id 53 in 75.
My prediction was right that the accuracy is chance (1/62 = 0.016; the recorded
chance_acc 1/31 is wrong for this generator) and that the loss sits near ln(62); it was
wrong that predictions would spread over most data ids: the argmax collapses onto a few.
