# Phase 2 prediction: cost figures (no training)

Written and committed before any Phase 2 run (rule 1). Standing approval (D6); budget ~20
GPU-min for 2.1, 2.2 runs on the local CPU. Small matched configs (D1).

## 2.1 Compute and memory scaling (Kaggle T4)

One forward+backward at random init, lengths 128-4096, batch 2 (one batch size for every
model and length), 2 warmup + 5 timed repeats, reset_peak_memory_stats before each.
Arms: (a) Transformer, naive attention (materialises T x T); (b) the same with torch SDPA;
(c) TCN; (d) mamba_ssm. Memory slope is fitted on peak minus pre-pass allocation (the part
that scales with length); time on the median repeat. The Transformer is built with
max_seq_len 4096, which adds positional-table parameters only (106,496 -> 425,984); per D5
this is disclosed, and it does not change compute per token.

**Prediction (plan2):** (a) memory slope rises toward 2 at long lengths and is below 2
over the full range; (b) memory slope near 1, time slope toward 2; (c) and (d) near 1 for
both memory and time. Mamba training memory is not flat.
**Gate:** top-three-length slopes within 0.25 of: (a) memory 2; (b) memory 1, time 2;
(c) memory 1, time 1; (d) memory 1, time 1. Other slopes reported only. My caveat: at
batch 2 and ~450k parameters, fixed per-pass costs may flatten the time slopes at short
lengths; the gate uses the top three lengths for that reason.

## 2.2 Streaming inference state (local CPU)

Bytes carried between tokens during one-token-at-a-time generation, contexts 128-4096,
batch 1, float32, measured by steppers that reproduce each model's full forward
(tests/test_streaming.py) and computed from shapes. Mamba via the custom model's per-step
loop (same state shapes as mamba_ssm).
**Prediction (plan2):** KV cache linear in context; TCN constant once context exceeds the
receptive field (769); Mamba constant.
**Gate:** analytic and measured agree within 5%.

## 2.2 outcome (appended after the run; prediction above unchanged)

Local CPU, ~10 s. **Gate PASS**: analytic and measured identical at every context. KV cache
linear (319,488 bytes at 128 -> 10,223,616 at 4096); Mamba constant (75,392); TCN constant
from context 384 on (301,056). The prediction held, with one correction to its wording: the
TCN's carried state saturates at its largest single layer window ((kernel - 1) x max
dilation = 384), not at the receptive field (769), which is the sum over layers. Figure:
streaming_state.png.
