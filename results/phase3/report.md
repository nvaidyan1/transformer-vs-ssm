| model | min_gap | seeds | final val_acc mean [min, max] |
|---|---|---|---|
| TCN, RF 127 | 32 | 2 | 0.936 [0.923, 0.949] |
| TCN, RF 127 | 64 | 2 | 0.258 [0.158, 0.358] |
| TCN, RF 127 | 96 | 2 | 0.012 [0.008, 0.017] |
| TCN, RF 127 | 160 | 2 | 0.013 [0.007, 0.020] |
| TCN, RF 127 | 256 | 2 | 0.015 [0.008, 0.021] |
| TCN, RF 127 | 512 | 2 | 0.014 [0.008, 0.021] |
| TCN, RF 769 | 32 | 2 | 0.249 [0.238, 0.260] |
| TCN, RF 769 | 64 | 2 | 0.409 [0.289, 0.529] |
| TCN, RF 769 | 96 | 2 | 0.201 [0.199, 0.203] |
| TCN, RF 769 | 160 | 2 | 0.205 [0.175, 0.234] |
| TCN, RF 769 | 256 | 2 | 0.124 [0.062, 0.187] |
| TCN, RF 769 | 512 | 2 | 0.014 [0.008, 0.021] |
| Transformer | 64 | 2 | 0.984 [0.976, 0.993] |
| Transformer | 256 | 2 | 0.995 [0.995, 0.995] |
| Mamba (mamba-ssm) | 64 | 2 | 0.997 [0.995, 0.998] |
| Mamba (mamba-ssm) | 256 | 2 | 0.521 [0.049, 0.993] |

Gate (tcn_short <= 2x chance at every gap above RF 127, > 0.9 at gap 32): **INCOMPLETE**

D3 extensions needed: 12
- tcn_short_g32_s2: ambiguous: final 0.771 in [0.2, 0.9]; unconverged: +0.555 over the last 1000 steps
- tcn_short_g64_s0: ambiguous: final 0.358 in [0.2, 0.9]; unconverged: +0.265 over the last 1000 steps
- tcn_short_g64_s1: unconverged: +0.107 over the last 1000 steps
- tcn_small_g160_s0: unconverged: +0.060 over the last 1000 steps
- tcn_small_g160_s1: ambiguous: final 0.234 in [0.2, 0.9]; unconverged: +0.096 over the last 1000 steps
- tcn_small_g256_s1: unconverged: +0.091 over the last 1000 steps
- tcn_small_g32_s0: ambiguous: final 0.260 in [0.2, 0.9]; unconverged: +0.086 over the last 1000 steps
- tcn_small_g32_s1: ambiguous: final 0.238 in [0.2, 0.9]; unconverged: +0.074 over the last 1000 steps
- tcn_small_g64_s0: ambiguous: final 0.289 in [0.2, 0.9]; unconverged: +0.102 over the last 1000 steps
- tcn_small_g64_s1: ambiguous: final 0.529 in [0.2, 0.9]; unconverged: +0.327 over the last 1000 steps
- tcn_small_g96_s0: unconverged: +0.062 over the last 1000 steps
- tcn_small_g96_s1: ambiguous: final 0.203 in [0.2, 0.9]; unconverged: +0.068 over the last 1000 steps
