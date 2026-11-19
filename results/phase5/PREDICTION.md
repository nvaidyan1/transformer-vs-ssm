# Phase 5 prediction: filtering experiments for Post 3

Written and committed before any 5.2/5.3 run (rule 1). Standing approval (D6). CLAIM_MAP
rows: Post 3 stationary tie, switching gap, effective gain vs K_t.

## 5.1 (done; gate passed)

`src/tasks/filtering.py`: random-walk latent (q = 0.01, x_0 ~ N(0, 1)); tasks (a) stationary
(r = 1.0), (b) switching (r in {0.05, 5.0}, geometric blocks of mean 32, flag = high noise),
(c) dropout (r = 0.1; with probability 0.3 the observation is pure N(0, 4) noise, flagged).
Exact Kalman filter (K = 0 on dropped steps) and best constant-gain filter (1-D search).
tests/test_filtering.py: (a) steady-state gain matches closed-form Riccati exactly and the
steady-state MSE within 1.1%; Kalman beats the best constant gain by 78% on (b) and 769% on
(c) (>= 20% required). Baseline MSEs (256 x 256 held-out): stationary Kalman 0.098 vs
constant 0.106; switching 0.097 vs 0.173; dropout 0.037 vs 0.322. 5.1 gate: PASS.

## 5.2 train small models

Regression mode (src/train_filter.py, src/models/regression.py): linear input projection,
linear head, MSE on the latent at every step. Models (d_model 32): selective Mamba (custom
blocks), Mamba ablation (input_independent=True), GRU; 1 layer, then 2. seq_len 256, batch
32, AdamW lr 1e-3 (the locked Mamba lr) with the 20,000-step schedule, 3000 steps, seeds 0-2,
5 GPU-min cap per run (plan2). Kaggle GPU instead of CPU: the local CPU ran at 5-15 s per
step under load, so CPU was not viable; this changes nothing in the method.

**Prediction (plan2):** on (a) all three reach the Kalman MSE within 10% and the ablation ties
the selective model. On (b) selective Mamba and the GRU approach the Kalman MSE and the
ablation sits closer to the constant-gain baseline. On (c): **uncertain**: a non-selective
model can suppress a flagged input without changing its forgetting, so the ablation may do
well.
**Caveat (mine, before any run):** the ablation removes input-dependent Delta, B and C but
keeps Mamba's input-dependent output gate silu(z) and the input-dependent conv path. A 3-step
smoke test already showed the ablation's effective gain differing by regime (0.51 low noise,
0.20 high noise) through the gate. So on (b) the ablation may track the regime too, and gate
(b) could fail for that reason; per plan2, Post 3 would then be written around it.
**Gate 5.2:** (a) ablation mean MSE within 10% of selective; (b) ablation MSE at least 15%
worse than selective on 3/3 seeds (paired), at 1 layer.

## 5.3 does the learned selection look like a gain?

On 32 held-out switching sequences: effective gain = d estimate_t / d y_t (autograd), Pearson
r with the Kalman K_t pooled over steps and sequences, for every switching model; mean Delta
per layer and channel split by regime (selective Mamba); scatter, regime-split Delta plot,
example trace.
**Prediction (plan2):** selective Mamba's effective gain correlates with K_t (r > 0.7) and is
higher in the low-noise regime; the ablation's varies much less across regimes. (By the caveat
above, the ablation may also vary.)
**Gate 5.3:** r > 0.7 for selective Mamba (1 layer) on 3/3 seeds.
5.4 (pendulum) is optional and only if 5.2 and 5.3 pass; not run here.

**Budget:** 54 runs at <= 5 GPU-min each plus the 5.3 analysis; the kernel stops at 4.5 h.
