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
