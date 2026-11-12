# Schedule follow-ups (author, 2026-11-12): prediction

Written and committed before any run (rule 1). Budget: 30 GPU-min, mamba_ssm ablated
(`--ablate`, `mamba_ssm_small.yaml`, lr 1e-3), report only. Every run here uses
`--schedule_steps 20000` (protocol change, results/LOG.md).

(a) Schedule lengths per arm: see the table in results/LOG.md (from the saved records).

(b) 1.2 seq_len 256 ablated arm rerun (it used 4,000 vs the pilots' 20,000): seeds 0-2,
`--steps 4000` (stop at CAP), early stop as usual. Output
`results/phase1_2/seq256_nd4_sched20k/`. The setting's gate is re-evaluated on these runs
with the existing selective pilots. **Prediction:** stays at chance (~0.016) on every
seed; gate PASS.

(c) Long-budget check: ablated, n_data 4, random positions, seed 0, at seq_len 64 and
seq_len 256, the full 20,000 steps (no early stop). Output
`results/phase1_sched_followups/longbudget/`. **Prediction (author):** stays below 0.2.

(d) 1.1d ablated n_data 2 and 3 rerun (1.1d used 4,000 for both arms): seeds 0-2,
`--steps 4000`, early stop as usual. Output `results/phase1_1d/sched20k/`.
**Prediction (mine):** both improve on the 4,000-schedule values (0.715 and 0.345),
since the lr stays near peak for the whole run (cf. 1.1c item 4), and the fall with
n_data remains (n_data 3 below n_data 2, both below 0.95).

Order: (b), (d), then (c) seq_len 64 and seq_len 256 last (longest); per-item deadlines
inside one 29-minute kernel. A cut run keeps its curve so far.
