# Saved models for the W1 and W2 figures

Written before the run. CLAIM_MAP rows: Post 1 Figs 3-4 (attention maps from the trained
Selective Copying Transformer) and Post 2 Fig 4 (Delta trace, illustration only). No gate;
these produce weights for figures, not reported numbers.

selective_copy, n_data 4, seq_len 64, seed 0, `--schedule_steps 20000`, early stop at
val_acc > 0.99 for 3 evals, at most 20,000 steps, `--save_model`:
- transformer_small.yaml (attention maps drawn locally from the saved weights)
- mamba_ssm_small.yaml (weights copied into the custom model locally, which returns Delta)

Expected: both early-stop above 0.99 (they did in 0.3 and 1.2).

Outcome (lane A, commit 421845b): transformer_s0 early-stopped at step 4500 (0.995);
mamba_ssm_s0 at step 3000 (0.996). Weights saved as *.pt (1.8 MB each).
