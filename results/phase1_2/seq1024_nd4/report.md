## seq1024_nd4

| arm | seed | schedule | steps-to-0.95 | steps run | early stop | final val_acc | wall s |
|---|---|---|---|---|---|---|---|
| mamba_ssm selective | 0 | 20000 | 3250 | 4750 | True | 0.9932 | 460 |
| mamba_ssm selective | 1 | 20000 | 2750 | 5000 | True | 0.9961 | 502 |
| mamba_ssm selective | 2 | 20000 | 5750 | 7000 | True | 0.9961 | 703 |
| transformer | 0 | 20000 | 2500 | 6500 | False | 0.9932 (partial) | 1674 |

CAP = 9000

Gate: **INCOMPLETE**
