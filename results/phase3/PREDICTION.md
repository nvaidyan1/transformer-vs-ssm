# Phase 3 prediction: what the TCN pays (reach)

Written and committed before any Phase 3 run (rule 1). Standing approval (D6); scope D2;
CAP rule D3 (results/LOG.md). CLAIM_MAP row: Post 1 Fig 5 / "past that point, the locality
assumption costs".

## 3.1 (done before this file; gate passed)

`min_gap` option in selective_copy: data only in the first input_len - min_gap positions,
so every answer is at least min_gap after its source (property tests, all 6 gaps).
`configs/recall/tcn_short.yaml`: 6 layers, kernel 3, d_model 132, **receptive field 127**,
440,088 params (-1.9% vs 448,632). tests/test_tcn_receptive_field.py verifies by gradients:
output at t has exactly zero gradient w.r.t. inputs at t-k for every k >= RF, non-zero at
k = RF - 1, for tcn_short (127) and tcn_small (769). 3.1 gate: PASS.

## 3.2 the reach curve

selective_copy, n_data 4, random positions, min_gap in {32, 64, 96, 160, 256, 512},
seq_len = min_gap + 64, vocab 64. `--schedule_steps 20000`, CAP 4000, early stop at
val_acc > 0.99 for 3 evals. Arms (D2): tcn_short and tcn_small at all six gaps, seeds 0-2;
Transformer (transformer_small) and mamba_ssm at gaps 64 and 256, seeds 0-2. The two gaps:
64 is inside tcn_short's reach for every token (distances 64-126 < 127) and 256 is outside
it, and both fit the Transformer's locked 512-token config. Distances from source to answer
span min_gap to about min_gap + 62, so gap 96 straddles tcn_short's RF.

**Prediction (plan2):** tcn_short near ceiling below its RF and at chance above it;
Transformer and mamba_ssm unaffected; tcn_small (reach sufficient everywhere): no
prediction, reported as a test of fixed routing rather than reach.
Mine, additionally: tcn_short partial at gap 96; mamba_ssm solves both gaps within CAP; the
Transformer may not reach 0.95 by CAP 4000 at gap 256 (it needed 12,750-15,000 steps at
seq_len 256 in 1.2) and would then be extended under D3.

**Gate:** tcn_short at every gap above its RF (160, 256, 512) within 2x chance
(<= 2/62 = 0.032), AND tcn_short > 0.9 at the smallest gap (32). Final eval, 3-seed mean.
plan2 says "the two gaps above its RF", written for an RF near 128 before the gaps were
fixed; with RF 127 three gaps are above it, and all three are gated (the stricter reading).

**Budget:** ~3 GPU-h (CLAIM_MAP). The kernel stops at 3.5 h of training; cut runs keep
their curves. Extensions (D3) run in a follow-up kernel once the list is known.

## Run outcome (appended after the run; prediction above unchanged)

Kaggle T4 lane B, commit 4fedeac, 3.5 h of training (the cap), 179 tests passed. Seeds 0
and 1 complete for every cell (32 runs); seed 2 not run (deadline), one seed-2 run cut
(tcn_short_g32_s2, partial). The TCN is slower than estimated (~6.6 min per CAP-4000 run).
Gate: **INCOMPLETE** (needs 3 seeds). With 2 seeds: tcn_short 0.936 at gap 32 (> 0.9) and
0.012-0.015 at gaps 96-512 (below 2x chance, 0.032), i.e. on track to pass; 0.258 and rising
at gap 64. tcn_small (RF 769, no prediction) at CAP 4000: 0.25 / 0.41 / 0.20 / 0.21 / 0.12 /
0.01 at gaps 32-512, mostly still rising. Transformer 0.984 (gap 64) and 0.995 (gap 256);
mamba_ssm 0.997 (gap 64) and 0.521 at gap 256 (one seed 0.993, the other 0.049). 12 cells
meet the D3 extension rule (11 are TCN cells, each ~5x a CAP run).

## Gate completion, P3a (author approval, 2026-11-23; written before the run)

Seed 2 for the four tcn_short gate cells only (gaps 32, 160, 256, 512), same settings (CAP
4000, 20k schedule, early stop). The partial seed-2 gap-32 run is replaced by a full one.
Other seed-2 cells (P3b) and the 12 D3 extensions (P3c) are not run.
**Prediction:** gap 32 > 0.9 (seeds 0-1: 0.923, 0.949); gaps 160-512 at chance (<= 0.032);
gate PASS. If seed 2 at gap 32 lands in [0.2, 0.9] or is still rising, D3 calls for an
extension before the gate is read.
