# Phase 4 prediction: what Mamba pays (state capacity)

Written and committed before any Phase 4 run (rule 1). Standing approval (D6); CAP rule D3.
CLAIM_MAP row: Post 2 new section "what compression costs".

MQAR (src/tasks/mqar.py), n_pairs in {4, 8, 16, 32, 64}, n_queries = n_pairs, seq_len =
4 x n_pairs (the generator's minimum: "just long enough to hold them"), vocab 256 (127 keys,
chance 1/127; vocab 64 has only 31 keys and cannot hold 64 pairs). Arms: mamba_ssm at
d_model 32 / 64 / 128 (4 layers, d_state 16; 48,192 / 147,584 / 499,968 params) and a
Transformer matched to each (47,936 / 143,192 / 512,176; within 3%, positional table
counted, non-positional counts disclosed in configs/mqar/). Seeds 0-2.
`--schedule_steps 20000`, CAP 4000, early stop at val_acc > 0.99 for 3 evals; D3 extensions
in a follow-up kernel.

**Prediction (plan2):** the Transformer stays near ceiling across the sweep; Mamba accuracy
falls past a threshold number of pairs, and the threshold moves right as d_model grows.
Mine, additionally: d_state 16 x d_inner 64 for the smallest Mamba holds roughly 1,000
numbers per layer, so I expect its fall to start by 16-32 pairs.

**Gate:** for mamba_ssm d32, final accuracy at 64 pairs is at least 0.3 below its accuracy
at 4 pairs, while the Transformer matched to it (transformer_m32) drops by less than 0.1.
3-seed means, after any D3 extension.
If the gate fails because Mamba holds up: extend to 128 pairs once (plan2). That needs a
vocabulary of at least 257, so it would also need a logged config change; it is not done
unless the gate fails that way.

**Budget:** ~2-3 GPU-h (CLAIM_MAP). 90 runs; the kernel stops at 3 h of training.

## CAP-stage outcome (appended after the run; prediction above unchanged)

Kaggle T4 lane A2, commit 9c34434, ~2.1 GPU-h, 90/90 runs, 182 tests passed. Gate:
**INCOMPLETE**, both 4-pair gate cells await a D3 extension. At CAP 4000 no model solves
MQAR even at 4 pairs: mamba_ssm 0.51 / 0.55 / 0.65 (d32 / d64 / d128), every Transformer
0.26-0.27 (all sizes, all seeds), consistent with the long plateau MQAR is known for; the
task layout was checked (key-value blocks, then key-query blocks with teacher-forced values;
14 property tests). Accuracy falls with pairs for every model. 37 cells meet the D3 rule.
The "Transformer stays near ceiling" prediction cannot be assessed before the extensions.

## Gate completion, P4a (author approval, 2026-11-23; written before the run)

D3 extensions for the six gate cells only: mamba_ssm_d32 and transformer_m32 at 4 pairs,
seeds 0-2, rerun to 20,000 steps on the same 20k schedule (same seed, so the first 4000
steps repeat the CAP run up to GPU nondeterminism), early stop as usual. Output in
results/phase4/ext/. The 64-pair gate cells were flat near chance and final at CAP.
**Prediction:** the Transformer leaves its 0.26 plateau and approaches ceiling at 4 pairs
(Zoology-style MQAR is solvable by attention given steps); mamba_ssm_d32 rises above 0.51.
Gate: mamba_d32 drop (4 -> 64 pairs) >= 0.3 is very likely (0.51 already vs 0.03); the
Transformer condition (drop < 0.1) needs it near ceiling at 64 pairs too, which was 0.03 at
CAP and is not extended, so I expect the Transformer condition, and the gate, to FAIL unless
its 64-pair cell is final-and-low only because of the CAP. Recorded honestly: the gate as
designed compares an extended 4-pair cell with an unextended 64-pair cell.
P4a outcome (lane D, commit 8b4dcaa): both 4-pair gate cells ran the full 20,000 steps
without escaping their plateaus: transformer_m32 0.26 on every seed for all 20,000 steps
(0.25-0.27 throughout); mamba_ssm_d32 rose to ~0.52 by step 2000 and stayed there.
**Gate FAIL**: Mamba drop 0.495 (>= 0.3, ok); Transformer drop 0.232 (needs < 0.1). My
prediction that the Transformer would leave its plateau was wrong. The fixed plateaus near
1/4 (Transformer) and 1/2 (Mamba) at 4 pairs, unchanged for 16,000 extra steps, are
suspicious for this small task (16 tokens, 4 queries); I have not investigated (gate failed:
stop and report). Per-slot accuracy would show whether each model answers a fixed subset of
query slots; that would need the author's approval.
