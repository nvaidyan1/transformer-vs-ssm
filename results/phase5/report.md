## 5.2 MSE against exact baselines

| task | model | layers | MSE mean [min, max] | / Kalman | / constant gain |
|---|---|---|---|---|---|
| stationary | selective Mamba | 1 | 0.1030 [0.1028, 0.1032] | 1.01 | 0.93 |
| stationary | Mamba ablation | 1 | 0.1032 [0.1029, 0.1038] | 1.02 | 0.93 |
| stationary | GRU | 1 | 0.1030 [0.1026, 0.1035] | 1.01 | 0.93 |
| stationary | selective Mamba | 2 | 0.1050 [0.1021, 0.1085] | 1.03 | 0.94 |
| stationary | Mamba ablation | 2 | 0.1040 [0.1025, 0.1049] | 1.03 | 0.93 |
| stationary | GRU | 2 | 0.1023 [0.1019, 0.1030] | 1.01 | 0.92 |
| switching | selective Mamba | 1 | 0.0996 [0.0967, 0.1034] | 1.06 | 0.56 |
| switching | Mamba ablation | 1 | 0.1102 [0.1083, 0.1116] | 1.18 | 0.62 |
| switching | GRU | 1 | 0.0969 [0.0953, 0.0999] | 1.03 | 0.55 |
| switching | selective Mamba | 2 | 0.0986 [0.0960, 0.1007] | 1.05 | 0.56 |
| switching | Mamba ablation | 2 | 0.1143 [0.1036, 0.1224] | 1.22 | 0.65 |
| switching | GRU | 2 | 0.0965 [0.0958, 0.0973] | 1.03 | 0.55 |
| dropout | selective Mamba | 1 | 0.0388 [0.0382, 0.0394] | 1.04 | 0.11 |
| dropout | Mamba ablation | 1 | 0.0392 [0.0387, 0.0399] | 1.05 | 0.11 |
| dropout | GRU | 1 | 0.0377 [0.0377, 0.0378] | 1.01 | 0.11 |
| dropout | selective Mamba | 2 | 0.0388 [0.0382, 0.0395] | 1.04 | 0.11 |
| dropout | Mamba ablation | 2 | 0.0388 [0.0385, 0.0391] | 1.04 | 0.11 |
| dropout | GRU | 2 | 0.0380 [0.0377, 0.0385] | 1.02 | 0.11 |

Gate 5.2 ((a) stationary ablation within 10%; (b) switching ablation >= 15% worse on 3/3 seeds; 1 layer): **FAIL**
- (a) relative difference +0.3% (ok)
- (b) ablated / selective by seed: s0 1.12, s1 1.08, s2 1.12 (MISS)

## 5.3 effective gain vs Kalman gain (switching, held-out)

Kalman gain: low-noise 0.379, high-noise 0.035.

| model | Pearson r with K_t | mean gain, low noise | mean gain, high noise |
|---|---|---|---|
| switching_gru_L1_s0 | 0.980 | 0.379 | 0.036 |
| switching_gru_L1_s1 | 0.975 | 0.374 | 0.034 |
| switching_gru_L1_s2 | 0.980 | 0.385 | 0.035 |
| switching_gru_L2_s0 | 0.991 | 0.387 | 0.037 |
| switching_gru_L2_s1 | 0.991 | 0.362 | 0.033 |
| switching_gru_L2_s2 | 0.969 | 0.383 | 0.035 |
| switching_mamba_L1_s0 | 0.980 | 0.385 | 0.035 |
| switching_mamba_L1_s1 | 0.971 | 0.400 | 0.036 |
| switching_mamba_L1_s2 | 0.979 | 0.396 | 0.033 |
| switching_mamba_L2_s0 | 0.984 | 0.367 | 0.031 |
| switching_mamba_L2_s1 | 0.983 | 0.389 | 0.035 |
| switching_mamba_L2_s2 | 0.985 | 0.382 | 0.037 |
| switching_mamba_ablated_L1_s0 | 0.984 | 0.470 | 0.036 |
| switching_mamba_ablated_L1_s1 | 0.979 | 0.456 | 0.043 |
| switching_mamba_ablated_L1_s2 | 0.989 | 0.424 | 0.036 |
| switching_mamba_ablated_L2_s0 | 0.981 | 0.423 | 0.041 |
| switching_mamba_ablated_L2_s1 | 0.969 | 0.397 | 0.040 |
| switching_mamba_ablated_L2_s2 | 0.977 | 0.400 | 0.039 |

Gate 5.3 (selective Mamba, 1 layer, r > 0.7 on 3/3 seeds): **PASS**
- Delta (layer 1): low-noise mean 0.0403 (channel std 0.0557), high-noise 0.0233 (0.0370); channels higher in low noise 64%
- Delta (layer 1): low-noise mean 0.1282 (channel std 0.1656), high-noise 0.0354 (0.0364); channels higher in low noise 61%
- Delta (layer 1): low-noise mean 0.0456 (channel std 0.0751), high-noise 0.0277 (0.0480); channels higher in low noise 55%
