# Phase 0.2 follow-up (amended): weight-copied pair

Written and committed before the run (rule 1).

## What runs

Author's amendment to plan2.md 0.2, item 2: one weight-copied pair in the cell
`emb_init_std=1.0, decay_1d=False`, seed 0 (deliberately not the failing cell).
`out_proj_depth_scale=True`, as in every 2x2 cell. The official model (mamba_ssm) is
built at seed 0 exactly as in the 2x2; its weights are copied into the custom model; both
train for 2000 steps in lockstep with identical AdamW settings on the identical batch at
each step; both are evaluated on the same fixed validation set every 250 steps.
Script: `scripts/phase0_weight_copy.py`. Single seed by the author's amendment (this
overrides rule 2 for this item). Do not investigate further whatever the outcome.

Item 1 (done before this run, CPU, vendored official constructor): out_proj depth
scaling was True for both archs in every 2x2 record; no parameter's step-0 std differs by
more than 10% between custom and official at either emb_init_std
(`init_std_vendored.json`). The kernel repeats it with the real mamba-ssm module.

## Prediction

No numeric gate was set for this item; the outcome is descriptive.
1. Step-0 logits agree to float tolerance (< 1e-3) after the copy.
2. The two accuracy curves coincide at the first evals and separate later, since recap.md
   section 6 shows a small divergence from identical weights that compounds.
3. The official curve approximately reproduces the 2x2 official seed-0 run (0.193 at step
   2000), since init and batches are the same; CUDA nondeterminism may perturb it.
4. At step 2000 custom is below official (in the 2x2 at emb 1.0, official was ahead in
   5 of 6 seed pairs). No prediction on the size of the gap.

## Budget

~18 GPU-min left of the second approved budget. Pair run capped at 13 min by the kernel;
a cut run keeps its curve (record rewritten every eval).

## Outcome (appended after the run; prediction above unchanged)

Kaggle T4, commit f0611ee, ~12 GPU-min (second budget: ~54 of 60 used). 103 tests passed.
Real mamba-ssm init-std check: no tensor differs by >10% at either emb_init_std.
1. Step-0 max |logit diff| 3.05e-05: as predicted.
2. Curves coincide through step 1000 (|diff| <= 0.001) and separate only slightly later:
   max |diff| 0.0049 at step 2000. Smaller separation than anticipated.
3. Official 0.196 at step 2000 vs 0.193 in the 2x2 seed-0 run: as predicted.
4. Custom below official at step 2000 by 0.005 (0.191 vs 0.196): sign as predicted,
   size negligible against the seed-to-seed spread in the 2x2 (0.16 to 0.68).
Not investigated further, per the amendment.
