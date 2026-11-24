# Results index

One directory per rung. Each has PREDICTION.md (committed before the run), raw JSON, the
Kaggle log, and a report generated from the JSON by a script in `scripts/`.

| Directory | plan2.md item | Outcome |
|---|---|---|
| phase0_1_smoke | 0.1 one harness | gate PASS |
| phase0_2x2 | 0.2 the 2x2 | gate FAIL as written; closed by the weight-copy follow-up (cfab726) |
| phase0_weight_copy | 0.2 follow-up (amended) | from identical weights, custom tracks official within 0.005 |
| phase0_3_lr | 0.3 lr check and Transformer control | control gate PASS (mean 0.986) |
| phase1_1 | 1.1 Selective Copying, seq_len 64 | gate PASS |
| phase1_1b | 1.1b positive control | gate FAIL (ablated 0.258 on fixed positions) |
| phase1_1c | 1.1c what the ablated model can learn | gate FAIL as written; post-hoc amendment clears the ablation (LOG.md) |
| phase1_1d | 1.1d items the ablated model can carry (Post 2 figure) | no gate; ablated 1.00 / 0.72 / 0.35 / 0.02 at n_data 1-4, selective >= 0.99 |
| phase1_2a | 1.2a ablation on the official model | gate PASS |
| phase1_sched_followups | schedule follow-ups (author, 2026-11-12) | at seq_len 64 the ablated model reaches 0.978 by 20k steps (slower, not incapable); at 256 it stays at 0.08 |
| phase2 | 2.1 cost scaling, 2.2 streaming state | 2.1 gate FAIL (memory slopes all as predicted; 3 time slopes overhead-bound); 2.2 PASS |
| figure_models | saved models for W1/W2 figures | both trained (0.995, 0.996) |
| phase3 | 3.2 TCN reach curve | gate PASS (3 seeds): tcn_short 0.93 at gap 32, chance at every gap beyond its RF 127 |
| phase4 | 4.1 MQAR capacity | gate FAIL: at 4 pairs the Transformer stays at 0.26 and Mamba at 0.52 for 20,000 steps (unexplained plateaus) |
| phase5 | 5.1-5.3 filtering (Post 3) | 5.1 PASS; 5.2 FAIL on (b): ablation only 8-12% worse on switching noise (needs 15%); 5.3 post hoc: every model's gain tracks K_t (r ~0.98), ablation included |
| phase1_2 | 1.2 lengthen the gate | seq_len 256: gate PASS (also on the 20k-schedule rerun); seq_len 1024: pilot done, CAP 9000, ablated arm pending |

Log of changes that affect how results are read: [LOG.md](LOG.md).

**Erratum (2026-11-12):** results written before this date store `chance_acc` = 1/31 for selective_copy; the correct value is 1/62 (LOG.md).
