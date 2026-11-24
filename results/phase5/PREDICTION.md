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

## 5.2 outcome (appended after the run; prediction above unchanged)

Kaggle T4 lane C, commit 48e3ac4, ~2.4 h, 54/54 runs; 18 Mamba runs (selective and
ablated, 2 layers and some 1-layer) hit the 5-min cap at ~2850 of 3000 steps
(stopped_by_time_cap, valid by plan2's cap). **Gate 5.2: FAIL** on (b).
- (a) stationary, 1 layer: ablation within 0.3% of selective; all three models within 3% of
  the Kalman MSE. As predicted.
- (b) switching, 1 layer: ablation / selective MSE = 1.12, 1.08, 1.12 by seed; the gate
  needs >= 1.15 on 3/3. Selective Mamba 1.06x Kalman, GRU 1.03x, ablation 1.18x; the
  constant-gain filter is 1.79x. The ablation is worse, but far closer to Kalman than to the
  constant-gain baseline, so "the ablation sits closer to the constant-gain baseline" is
  wrong. Consistent with the caveat recorded above (the ablation keeps the input-dependent
  output gate and conv path, through which the flag can act).
- (c) dropout: all three within 5% of Kalman, including the ablation (plan2 marked this
  uncertain; the ablation does well, as plan2 anticipated could happen).
Per plan2 5.2: stopped and reported; Post 3 is to be written around this (W3).

5.3 did not run: the analysis step crashed on its first GRU model (cuDNN refuses RNN
backward in eval mode; a GPU-only failure the CPU tests could not catch). Fixed in
src/filter_analysis.py after the run. No 5.3 numbers exist. Running it now would be a
descriptive analysis after a failed 5.2 gate and needs the author's decision.

## 5.3 run as a post-hoc descriptive analysis (author approval, 2026-11-23)

Written before the run. Gate 5.2 failed, so per plan2 Phase 5 stopped; the author approved
running 5.3 anyway as a descriptive analysis for W3 ("where selection lives in the block").
It is labelled post hoc wherever used. Same script and saved models (results/phase5/models,
the 18 switching models), with the cuDNN fix. The 5.3 gate (r > 0.7 for selective Mamba,
3/3 seeds) is still computed and reported, but it no longer decides anything.
**Expectation (mine):** selective Mamba's effective gain correlates with K_t and is higher in
the low-noise regime; the ablation also varies by regime (through the output gate), less
than selective; the GRU tracks K_t closely (its MSE is 1.03x Kalman).
