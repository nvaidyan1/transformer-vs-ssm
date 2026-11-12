# Phase 1.1d prediction: items the ablated model can carry

Written and committed before any 1.1d run (rule 1). Added by the author, 2026-11-12.
Budget: 20 GPU-min. No gate; this is a figure for Post 2.

selective_copy, random positions, seq_len 64, vocab 64, n_data in {1, 2, 3, 4}.
mamba_ssm selective and mamba_ssm ablated (`--ablate`), locked recipe
(`mamba_ssm_small.yaml`, lr 1e-3), seeds 0-2, CAP 4000, early stop at val_acc > 0.99 for
3 consecutive evals; `--dump_slot_acc`. Chance 1/62. 24 runs, one at a time, ordered
seed-major (seed 0 at every n_data and arm first) so a deadline cut leaves whole seeds.

**Prediction (author):** selective >= 0.95 at every n_data; ablated solves n_data = 1
and falls as n_data grows.

Budget note: at ~60-85 steps/s for mamba_ssm at seq_len 64 plus ~10 s per process start,
24 runs fit in ~17-20 min if the selective runs early-stop; the kernel enforces a
19.5-minute deadline for this item.

## Outcome (appended after the run; prediction above unchanged)

Kaggle T4, commit 3242c64, ~18 GPU-min of 20. All 24 runs complete.
Selective: 1.000 / 0.996 / 0.999 / 0.998 (n_data 1-4, 3-seed means; every seed >= 0.994).
Ablated: 1.000 / 0.715 [0.566, 0.891] / 0.345 [0.234, 0.438] / 0.017 [0.014, 0.019].
Both parts of the prediction hold. Per slot, the ablated model does best on the last data
token (the shortest carry): n_data 2 slots 0.49 / 0.94, n_data 3 0.09 / 0.14 / 0.80; at
n_data 4 every slot is at chance (0.01-0.02), unlike the fixed-position n_data 4 run in
1.1c (last slot 0.988). Figure: items_carried.png.
