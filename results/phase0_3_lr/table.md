| arch | params | lr | val_acc @4000 | peak val_acc | stability | picked |
|---|---|---|---|---|---|---|
| transformer | 452,296 | 0.0003 | 0.9756 | 0.9863 | stable |  |
| transformer | 452,296 | 0.001 | 1.0000 | 1.0000 | stable | **yes** |
| transformer | 452,296 | 0.003 | 0.9355 | 0.9355 | stable |  |
| tcn | 450,212 | 0.0003 | 0.7939 | 0.7939 | stable | **yes** |
| tcn | 450,212 | 0.001 | 0.2510 | 0.2510 | stable |  |
| tcn | 450,212 | 0.003 | 0.2637 | 0.2637 | stable |  |
| mamba_ssm | 448,632 | 0.0003 | 0.9795 | 0.9795 | stable |  |
| mamba_ssm | 448,632 | 0.001 | 0.9990 | 1.0000 | stable | **yes** |
| mamba_ssm | 448,632 | 0.003 | 1.0000 | 1.0000 | stable |  |

- transformer: picked lr = 0.001
- tcn: picked lr = 0.0003
- mamba_ssm: picked lr = 0.001
