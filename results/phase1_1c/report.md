0. Non-data positions: one blank token (id 0). The generator starts from np.full(seq_len, NOISE_ID) and overwrites only the data tokens and the marker; there are no random noise tokens.

| run | what | steps run | early stop | final val_acc | per-slot accuracy |
|---|---|---|---|---|---|
| 1 | custom ablated, fixed, n_data=4 (1.1b re-run) | 3000 | False | 0.2578 | 0.016, 0.016, 0.012, 0.988 |
| 2 | mamba_ssm ablated, fixed, n_data=1 (gate) | 3000 | False | 0.3516 | 0.352 |
| 3 | mamba_ssm ablated, random, n_data=1 | 2500 | True | 1.0000 | 1.000 |
| 4 | mamba_ssm ablated, fixed, n_data=2 | 3000 | False | 0.6895 | 0.379, 1.000 |

Gate (run 2 >= 0.9): **FAIL**. Chance is 1/62 = 0.0161.
