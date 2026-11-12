| arm (mamba_ssm, seed 0) | steps-to-0.95 | steps run | early stop | final val_acc |
|---|---|---|---|---|
| selective | 1500 | 2750 | True | 0.9990 |
| ablated | not reached | 3000 | False | 0.0137 |

Gate (selective >= 0.95 AND ablated < 0.2): **PASS**
