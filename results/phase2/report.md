## 2.1 Compute and memory scaling

| model | params | memory slope full / top-3 | time slope full / top-3 | OOM at |
|---|---|---|---|---|
| Transformer, naive attention | 825,032 | 1.76 / 1.92 | 1.00 / 1.81 | - |
| Transformer, SDPA | 825,032 | 1.00 / 1.01 | 0.74 / 1.63 | - |
| TCN | 450,212 | 0.97 / 0.82 | 0.25 / 0.60 | - |
| Mamba (mamba-ssm) | 449,752 | 0.99 / 1.00 | 0.23 / 0.50 | - |

Gate (top-3 slopes within 0.25 of prediction): **FAIL**
- Transformer, naive attention memory: predicted 2.0, measured 1.92 ok
- Transformer, SDPA memory: predicted 1.0, measured 1.01 ok
- Transformer, SDPA time: predicted 2.0, measured 1.63 MISS
- TCN memory: predicted 1.0, measured 0.82 ok
- TCN time: predicted 1.0, measured 0.60 MISS
- Mamba (mamba-ssm) memory: predicted 1.0, measured 1.00 ok
- Mamba (mamba-ssm) time: predicted 1.0, measured 0.50 MISS

## 2.2 Streaming inference state (bytes carried to the next token)

| context | Transformer KV cache | TCN | Mamba state |
|---|---|---|---|
| 128 | 319,488 | 175,616 | 75,392 |
| 256 | 638,976 | 250,880 | 75,392 |
| 512 | 1,277,952 | 301,056 | 75,392 |
| 1024 | 2,555,904 | 301,056 | 75,392 |
| 2048 | 5,111,808 | 301,056 | 75,392 |
| 4096 | 10,223,616 | 301,056 | 75,392 |

Gate (analytic vs measured within 5%): **PASS** (worst relative difference 0.00%)
