# Phase 1.1 prediction: Selective Copying gate at seq_len 64

Written and committed before Stage A (rule 1; plan2.md 1.1 as amended 2026-11-11, which
replaces the shared 60-minute budget for this task with per-stage budgets).

Task for every stage: `selective_copy`, `n_data=4`, `seq_len=64`, `vocab_size=64`
(chance 0.032), batch 32, eval every 250 steps on the fixed 8-batch validation set.
Early stop (Stages B, C): val_acc strictly above 0.99 for 3 consecutive evals.
Steps-to-0.95 = the first eval step with val_acc >= 0.95 (resolution 250 steps).
Note: `--steps` sets the cosine schedule length, so a run with a 20,000-step budget keeps
a higher lr for longer than the 4000-step runs of 0.3.

## Stage A: custom Mamba speed (max 10 GPU-min)

`configs/recall/mamba_small.yaml` (custom model, locked 0.3 settings: lr 3e-4,
emb_init_std 0.02), 200 timed steps after 20 warmup steps, in four configurations:
scan sequential/parallel x gradient checkpointing on/off. Then parity of the fastest
against the default (sequential, checkpointing on) on CUDA, the parity test files, and
steps/sec with 1, 2 and 3 processes sharing the GPU in the fastest configuration.
Script: `scripts/bench_custom_mamba.py`.

Prediction:
1. Checkpointing off is faster than on, by roughly 20-30% (no recomputed forward).
2. At T=64 on CUDA the parallel scan is faster than the sequential one: the sequential
   scan launches small kernels for each of 64 timesteps per layer and is launch-bound.
   (PLAN.md section 4 found the opposite on CPU and MPS; CUDA was not measured.)
   So the fastest is parallel + checkpointing off.
3. Parity in the fastest configuration: logits within 1e-5, relative gradient
   difference within 1e-4 (the scans differ only in float32 summation order).
4. Aggregate throughput rises with processes but sublinearly (the T4 kernel has few CPU
   cores): about 1.5-1.8x with 2 processes and 1.8-2.4x with 3.

## Stage B: mamba_ssm pilot (max 20 GPU-min)

`configs/recall/mamba_ssm_small.yaml` (locked recipe, lr 1e-3), selective, seeds 0-2,
up to 20,000 steps with the early stop.

Prediction: every seed reaches 0.95, in roughly 1000-2500 steps (the 0.3 lr check had
seed 0 at 0.91 by step 1250 and 0.98 by 1500), and early-stops within ~1000 steps after.
If no seed reaches 0.95 in 20,000 steps: stop and report, no Stage C. If some but not all
seeds reach it, CAP is undefined by the amendment: stop and ask.

## Stage C: custom selective and ablated (budget set at approval)

CAP = 1.5 x the slowest pilot seed's steps-to-0.95, rounded up to the next 1000, at most
20,000. Custom selective (`mamba_small.yaml`) and custom ablated (`--ablate`), seeds 0-2,
`--steps CAP`, same early stop, fastest Stage A configuration and best concurrency. The
projected GPU-minutes are posted for approval before launch.

Prediction:
1. The ablation stays near chance (well under 0.2) on every seed.
2. **Risk, recorded honestly:** the custom model runs its own reference recipe (lr 3e-4,
   emb_init_std 0.02), not mamba_ssm's (lr 1e-3, emb_init_std 1.0). In 0.2 that recipe
   reached only 0.09-0.35 by step 2000. A CAP derived from the faster mamba_ssm recipe
   will probably be ~3000-4000 steps, so I expect custom selective to fall short of 0.95
   on at least one seed: **gate FAIL more likely than PASS** under this configuration.
   The weight-copy result (results/phase0_weight_copy/) says such a failure would reflect
   the recipe, not the implementation. Whether to keep this configuration is the author's
   call at the Stage C approval; any change is logged and this prediction amended before
   launch.

Gate (unchanged): custom selective >= 0.95 on 3/3 seeds AND (selective - ablated) >= 0.5
at the final eval (mean over seeds for the difference). mamba_ssm pilot reported
alongside as the reference arm.
