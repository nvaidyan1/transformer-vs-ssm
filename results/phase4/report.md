| model | 4 pairs | 8 pairs | 16 pairs | 32 pairs | 64 pairs |
|---|---|---|---|---|---|
| mamba_ssm_d32 | 0.525 [0.51, 0.54] | 0.323 [0.29, 0.34] | 0.145 [0.12, 0.17] | 0.056 [0.05, 0.06] | 0.030 [0.03, 0.03] |
| mamba_ssm_d64 | 0.546 [0.52, 0.59] | 0.343 [0.33, 0.35] | 0.221 [0.21, 0.24] | 0.120 [0.11, 0.13] | 0.052 [0.05, 0.05] |
| mamba_ssm_d128 | 0.646 [0.58, 0.74] | 0.340 [0.33, 0.35] | 0.232 [0.23, 0.23] | 0.131 [0.09, 0.15] | 0.082 [0.08, 0.08] |
| transformer_m32 | 0.260 [0.26, 0.26] | 0.134 [0.13, 0.13] | 0.076 [0.07, 0.08] | 0.040 [0.03, 0.05] | 0.029 [0.02, 0.03] |
| transformer_m64 | 0.264 [0.26, 0.27] | 0.226 [0.19, 0.29] | 0.130 [0.11, 0.16] | 0.064 [0.06, 0.07] | 0.034 [0.03, 0.04] |
| transformer_m128 | 0.267 [0.26, 0.27] | 0.320 [0.31, 0.33] | 0.098 [0.09, 0.10] | 0.088 [0.06, 0.13] | 0.054 [0.04, 0.07] |

Gate (mamba_ssm_d32 drops >= 0.3 from 4 to 64 pairs; transformer_m32 drops < 0.1): **FAIL** (Mamba drop +0.495, Transformer drop +0.232)

D3 extensions needed: 31
- mamba_ssm_d128_p16_s0: ambiguous: final 0.233 in [0.2, 0.9]
- mamba_ssm_d128_p16_s1: ambiguous: final 0.232 in [0.2, 0.9]
- mamba_ssm_d128_p16_s2: ambiguous: final 0.232 in [0.2, 0.9]
- mamba_ssm_d128_p4_s0: ambiguous: final 0.619 in [0.2, 0.9]; unconverged: +0.088 over the last 1000 steps
- mamba_ssm_d128_p4_s1: ambiguous: final 0.740 in [0.2, 0.9]; unconverged: +0.115 over the last 1000 steps
- mamba_ssm_d128_p4_s2: ambiguous: final 0.579 in [0.2, 0.9]; unconverged: +0.060 over the last 1000 steps
- mamba_ssm_d128_p8_s0: ambiguous: final 0.329 in [0.2, 0.9]
- mamba_ssm_d128_p8_s1: ambiguous: final 0.347 in [0.2, 0.9]
- mamba_ssm_d128_p8_s2: ambiguous: final 0.345 in [0.2, 0.9]
- mamba_ssm_d32_p8_s0: ambiguous: final 0.294 in [0.2, 0.9]
- mamba_ssm_d32_p8_s1: ambiguous: final 0.336 in [0.2, 0.9]
- mamba_ssm_d32_p8_s2: ambiguous: final 0.339 in [0.2, 0.9]
- mamba_ssm_d64_p16_s0: ambiguous: final 0.236 in [0.2, 0.9]
- mamba_ssm_d64_p16_s1: ambiguous: final 0.219 in [0.2, 0.9]
- mamba_ssm_d64_p16_s2: ambiguous: final 0.208 in [0.2, 0.9]
- mamba_ssm_d64_p4_s0: ambiguous: final 0.529 in [0.2, 0.9]
- mamba_ssm_d64_p4_s1: ambiguous: final 0.594 in [0.2, 0.9]; unconverged: +0.054 over the last 1000 steps
- mamba_ssm_d64_p4_s2: ambiguous: final 0.516 in [0.2, 0.9]
- mamba_ssm_d64_p8_s0: ambiguous: final 0.347 in [0.2, 0.9]
- mamba_ssm_d64_p8_s1: ambiguous: final 0.351 in [0.2, 0.9]
- mamba_ssm_d64_p8_s2: ambiguous: final 0.332 in [0.2, 0.9]
- transformer_m128_p4_s0: ambiguous: final 0.271 in [0.2, 0.9]
- transformer_m128_p4_s1: ambiguous: final 0.269 in [0.2, 0.9]
- transformer_m128_p4_s2: ambiguous: final 0.260 in [0.2, 0.9]
- transformer_m128_p8_s0: ambiguous: final 0.335 in [0.2, 0.9]
- transformer_m128_p8_s1: ambiguous: final 0.320 in [0.2, 0.9]; unconverged: +0.081 over the last 1000 steps
- transformer_m128_p8_s2: ambiguous: final 0.305 in [0.2, 0.9]; unconverged: +0.082 over the last 1000 steps
- transformer_m64_p4_s0: ambiguous: final 0.262 in [0.2, 0.9]
- transformer_m64_p4_s1: ambiguous: final 0.263 in [0.2, 0.9]
- transformer_m64_p4_s2: ambiguous: final 0.267 in [0.2, 0.9]
- transformer_m64_p8_s0: ambiguous: final 0.289 in [0.2, 0.9]; unconverged: +0.054 over the last 1000 steps
