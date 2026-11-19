
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
