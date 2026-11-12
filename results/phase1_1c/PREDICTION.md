# Phase 1.1c prediction: what can the ablated model learn?

Written and committed before any 1.1c run (rule 1). Added by the author, 2026-11-11.
Budget: 30 GPU-min.

## 0. How the generator fills non-data positions

One blank token. `generate_selective_copy` starts from `np.full(seq_len, NOISE_ID)` with
NOISE_ID = 0 and writes the data tokens and the marker (id 1) over it: every non-data
position in the input region is the same token, id 0. There are no random noise tokens,
despite the module docstring's "field of noise tokens" wording.

## Setup

mamba_ssm arch with `--ablate` (input_independent=True; plan2.md 1.2a), locked recipe
`configs/recall/mamba_ssm_small.yaml` (lr 1e-3), seed 0, CAP 3000, early stop at val_acc
> 0.99 for 3 consecutive evals, seq_len 64, vocab 64. Fixed positions are a pure function
of the input region (`fixed_data_positions`): n_data=1 -> position 0 (marker at 62, so
the one item must be carried 62 steps, the longest possible); n_data=2 -> 0 and 60
(marker at 61); n_data=4 -> 0, 19, 39, 58 (marker at 59). Answer slot k reproduces the
k-th data token, so slot 4 is the one whose source is nearest its answer position
(4 steps), slot 1 the farthest (59 steps).

## Runs and predictions (the author's, recorded verbatim in substance)

1. Per-slot accuracy of the 1.1b fixed-position ablated run (custom model, n_data=4). No
   checkpoint exists, so it is re-run with the identical 1.1b command plus
   `--dump_slot_acc`. **Prediction:** one slot near 1.0, the others near chance (1/62).
   (Mine, additionally: the near-1.0 slot is slot 4, the shortest carry.)
2. Fixed positions, n_data=1. **Prediction:** ablated >= 0.9.
3. Random positions, n_data=1. **Prediction: uncertain**, recorded as uncertain.
4. Fixed positions, n_data=2. **Prediction:** ablated plateaus near 0.5.

My own note on run 2, recorded before the run: its single item sits at the farthest
possible position (62 steps). If the ablated model's plateau in 1.1b came from a short
memory rather than from carrying only one item, run 2 could fail for the same reason.

**Gate:** run 2 reaches >= 0.9 at the final eval.
- PASS: the 1.1 and 1.2a ablation results may be used in a post, worded as
  "input-dependent dynamics" (not "selection"), with the per-slot result and runs 2-4
  reported beside them; proceed to the ablated arms of 1.2.
- FAIL: stop and report. Do not modify the ablation.

## Budget

mamba-ssm install ~2.5 min and tests ~1 min; runs 2-4 on mamba_ssm ~1 min each; run 1
on the custom model ~8 min (it is last, so a deadline cut cannot cost the gate run). The
kernel enforces a 25-minute training deadline.

## Outcome (appended after the run; predictions above unchanged)

Kaggle T4, commit 4dee336, ~12 GPU-min of 30. 143 tests passed.

| run | final val_acc | per-slot | vs prediction |
|---|---|---|---|
| 1 custom ablated, fixed, n_data=4 | 0.258 | 0.016, 0.016, 0.012, 0.988 | as predicted: one slot near 1.0 (slot 4, the 4-step carry), others at chance |
| 2 mamba_ssm ablated, fixed, n_data=1 | 0.352 | 0.352 | **predicted >= 0.9: wrong.** Still rising at step 3000 (0.133, 0.281, 0.215, 0.230, 0.352 over the last five evals) |
| 3 mamba_ssm ablated, random, n_data=1 | 1.000 (early stop, 2500) | 1.000 | uncertain beforehand; solved |
| 4 mamba_ssm ablated, fixed, n_data=2 | 0.689 | 0.379, 1.000 | predicted plateau near 0.5: the near slot is 1.0, the far slot (60-step carry) 0.379 and rising |

**Gate FAIL** (run 2 needs >= 0.9). Stopped per the rule; the ablation is not modified.
Per the author's rule, the 1.1 and 1.2a ablation results are not usable in a post, and the
1.2 ablated arms do not run.
