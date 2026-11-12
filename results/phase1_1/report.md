## Stage A: custom Mamba training step (seq_len 64, batch 32)

| scan | grad_checkpoint | ms/step | steps/s | peak MiB |
|---|---|---|---|---|
| parallel | False | 154.2 | 6.48 | 2833 |
| sequential | False | 163.8 | 6.11 | 499 |
| parallel | True | 210.1 | 4.76 | 849 |
| sequential | True | 275.6 | 3.63 | 155 |

Parity, parallel / checkpointing False vs sequential / on (CUDA): max |logit diff| 5.96e-07, max relative grad diff 1.28e-06 over 47 tensors.

| processes | per-process steps/s | aggregate steps/s | speedup |
|---|---|---|---|
| 1 | 6.45 | 6.45 | 1.00x |
| 2 | 2.85 | 5.70 | 0.88x |

## Stage B: mamba_ssm pilot

| seed | steps-to-0.95 | steps run | early stop | final val_acc |
|---|---|---|---|---|
| 0 | 1500 | 3000 | True | 0.9902 |
| 1 | 1250 | 2500 | True | 0.9971 |
| 2 | 1250 | 3250 | True | 0.9990 |

CAP = ceil(1.5 x 1500 / 1000) x 1000 = **3000**

Stage C projection (6 runs to CAP, no early stop, +3 min setup):

- 1 process(es): 52 GPU-min
- 2 process(es): 58 GPU-min
