## seq256_nd4

| arm | seed | steps-to-0.95 | steps run | early stop | final val_acc | wall s |
|---|---|---|---|---|---|---|
| mamba_ssm selective | 0 | 1750 | 3000 | True | 0.9971 | 61 |
| mamba_ssm selective | 1 | 1750 | 2750 | True | 1.0000 | 61 |
| mamba_ssm selective | 2 | 2500 | 4000 | True | 0.9990 | 88 |
| transformer | 0 | 12750 | 20000 | False | 0.9795 | 507 |
| transformer | 1 | 15000 | 20000 | False | 0.9551 | 505 |
| transformer | 2 | 14250 | 16750 | True | 0.9932 | 424 |

CAP = 4000

Gate: **INCOMPLETE**
