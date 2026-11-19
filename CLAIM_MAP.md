# Claim-to-evidence map (2026-11-19)

One page: each post claim that needs a measurement, the one result that settles it, and its
status. Rule from now on: **no experiment runs unless a row below needs it.** Posts are not
edited here; W0/W1-W3 in plan2.md do that. Toy tasks on one T4 support "illustrates", not
"shows".

## Post 1: what attention assumes (Transformer pays compute, TCN pays reach)

| Claim / figure in the draft | Settled by | plan2 | Status |
|---|---|---|---|
| Fig 1: memory and time grow as n^2 with sequence length (slope placeholder) | log-log sweep 128-4096, naive vs SDPA attention | 2.1 | **not run** |
| Fig 2: TCN scales linearly where the Transformer is quadratic | same sweep, TCN | 2.1 | **not run** |
| Figs 3-4 (PENDING): attention re-routes per input | attention maps from the trained Selective Copying Transformer, two inputs | W1 (uses 1.1) | **needs a saved checkpoint** (trainer saves none yet) |
| Fig 5 + text (PENDING): the TCN's locality bet breaks past its reach | accuracy vs min_gap, RF line (tcn_short) | 3.1, 3.2 | **not run** |
| Quadratic cost is a worst-case conditional bound | citation (Duman Keles et al. 2023) | W0.1 | done |

## Post 2: what structure buys you (Mamba pays state capacity)

| Claim / figure in the draft | Settled by | plan2 | Status |
|---|---|---|---|
| Fig 2: training cost is linear (not flat) | 2.1 sweep, mamba_ssm | 2.1 | **not run** |
| "Mamba's memory is flat" (streaming state only) | carried state vs context, analytic = measured | 2.2 | **not run** |
| Fig 3: selectivity matters (replaces enwik8 bpc) | selective vs input-independent ablation | 1.1, 1.2 | **have it**: 0.99+ vs chance at seq 64 (3000 steps) and 256; with a matched 20k schedule the ablation reaches 0.978 at 64 (slower) and 0.08 at 256 (fails within budget) |
| Items-carried figure (author's Post 2 figure) | 1.1d | 1.1d | **mixed schedules**: n_data 2-3 ablated rerun on 20k, the other cells on 4k; rerun the rest or drop |
| Fig 4: Delta_t trace (illustration only) | Delta from a trained custom model on Selective Copying | W2 | **needs a saved checkpoint** |
| New section "what compression costs" | accuracy vs number of KV pairs, 3 Mamba sizes vs Transformer | 4.1 | **not run** |
| From-scratch model is a faithful reference; numbers from mamba-ssm | weight-copy parity, scan parity | 0.2, 1.2a | **have it** |

## Post 3: the same problem (does learned selection act like a Kalman gain?)

| Claim / figure in the draft | Settled by | plan2 | Status |
|---|---|---|---|
| Stationary problem: constant gain suffices, selectivity unnecessary | task (a): ablation ties selective, both near Kalman MSE | 5.1, 5.2 | **not run** |
| Switching noise: selectivity needed | task (b): ablation >= 15% worse MSE, 3/3 seeds | 5.2 | **not run** |
| Effective gain tracks K_t (replaces Figs 1-4) | Pearson r > 0.7, 3/3 seeds; regime-split Delta; example trace | 5.3 | **not run** |
| Hidden velocity is encoded (one sentence) | linear probe R^2 > 0.9 | 5.4 (optional) | optional |
| Lorenz (Fig 5) | none; appendix or cut | W3 | drop |

Decided in advance: if 5.2(b) or 5.3 fails, Post 3 is written around the failure, as plan2 W3
already says. Nothing else changes.

**Follow-ups (not run unless a row above starts needing them):** seq_len 1024 ablated arm and
n_data 16 (Phase 1 closes at 64 and 256); seq-length crossover sweep; a longer seq-256 ablation
run; Transformer seq-1024 completion.

## What it takes to cruise through the rest

**Operating rules.** One standing approval per phase (budget up front); stop only on a failed
gate or a budget overrun; no mid-phase amendments, new ideas go to Follow-ups; every new run
uses `--schedule_steps 20000`. Two Kaggle GPU sessions run in parallel; Phase 5 runs on the
local CPU alongside them.

**Code to write (local, tested, no GPU):**
1. Phase 2: `scripts/cost_scaling.py` (naive vs SDPA attention switch, peak memory and time,
   slopes) and `scripts/streaming_state.py` (analytic and measured carried state).
2. Trainer `--save_model`, plus dumps for attention maps (W1) and Delta traces (W2).
3. Phase 3: `min_gap` option in `selective_copy` with property tests; TCN receptive-field
   gradient test; `configs/recall/tcn_short.yaml` (RF ~128).
4. Phase 4: MQAR configs for mamba_ssm d_model 32/64/128 and parameter-matched Transformers.
5. Phase 5: `src/tasks/filtering.py` (three tasks, exact Kalman, best constant gain, Riccati
   tests); trainer regression mode (linear in/out, MSE); a GRU baseline; the effective-gain
   analysis (autograd) and plots.

**Compute, in order (estimates):**

| Order | Item | Where | Est. |
|---|---|---|---|
| 1 | Phase 2 (2.1 + 2.2) | 1 Kaggle kernel | ~20 GPU-min |
| 2 | Checkpoint runs for W1/W2 figures | same kernel | ~5 GPU-min |
| 3 | Phase 5.1-5.3 (54 tiny runs) | local CPU, background | 2-4 CPU-h |
| 4 | Phase 4 at CAP 4000 | Kaggle session A | ~2-3 GPU-h |
| 5 | Phase 3 | Kaggle session B | ~3 GPU-h reduced; full scope much more (TCN is the slow model) |
| 6 | 1.1d remaining cells on 20k (if kept) | either session | ~20 GPU-min |
| 7 | W0.2, W0.3, then W1-W3, F1 | text | after results |

**Author decisions needed before starting** are listed in plan2.md under "Cruise decisions".
