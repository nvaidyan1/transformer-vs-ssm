# Phase 1.2a prediction: the ablation switch on the official model

Written and committed before any 1.2a run (rule 1). Added by the author, 2026-11-11.
Budget: 20 GPU-min.

## The switch

`OracleMamba.forward(x, input_independent=True)`: every block runs
`ablated_mixer_forward`, the official mixer's slow path (use_fast_path=False code path)
with Delta, B and C replaced by learned constants `fixed_delta_pre`, `fixed_B`,
`fixed_C` (same shapes and init as the custom model's), still calling the official
`selective_scan_fn` (D skip and silu(z) gate inside it, as in the selective path).
The constants are created after every other parameter, so the rest of the model's init at
a given seed is unchanged (tested).

Tests (all must pass before training; the CUDA ones run on Kaggle):
1. With weights copied from a custom ablated model, outputs match within 1e-4 at T=64
   (CPU with the vendored reference scan; CUDA with the real kernels).
2. Ablated outputs do not change when x_proj and dt_proj are perturbed.

## Runs

`configs/recall/mamba_ssm_small.yaml` (locked recipe, lr 1e-3), selective_copy,
n_data 4, seq_len 64, vocab 64, seed 0, CAP 3000, early stop at val_acc > 0.99 for 3
consecutive evals. Arms: selective (config as is, fused fast path) and ablated
(`--ablate`).

**Prediction:** reproduces 1.1 on the official model: selective reaches > 0.99 and early
stops (the Stage B pilot seed 0 reached 0.95 at step 1500); ablated stays near 1/62.
**Gate:** selective >= 0.95 AND ablated < 0.2 at the final eval.

## Budget

mamba-ssm install ~2.5 min, tests ~1 min, two runs of at most 3000 steps (~2-3 min each
for mamba_ssm alone). The kernel enforces a 15-minute training deadline.

## Run log

- Attempt 1 (commit 006a0ba, ~2 GPU-min): all six 1.2a ablation tests passed, including
  both CUDA tests with the real kernels. The full suite then failed on one older test,
  `test_mamba_ssm_registered_and_lazy`, whose CUDA-only branch still expected 448,632
  parameters without the 1,120 new ablation-only ones. The kernel stopped before any
  training, as designed. Test fixed; no model or training change. Relaunched.
