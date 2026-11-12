# Phase 1.2 prediction: lengthen the Selective Copying gate (all arms on mamba_ssm)

Written before any 1.2 run (rule 1); extended before each later setting. plan2.md 1.2 as
amended by the author, 2026-11-11.

Settings, in order, stopping at the first failed gate: seq_len 256 (n_data 4), seq_len
1024 (n_data 4), seq_len 1024 (n_data 16). At each: pilot mamba_ssm selective and the
Transformer, 3 seeds each, up to 20,000 steps with early stop at val_acc > 0.99 for 3
consecutive evals; CAP = 1.5 x the slowest selective seed's steps-to-0.95 (rounded up to
the next 1000, max 20,000); then the mamba_ssm ablated arm (`--ablate`), 3 seeds, to CAP,
one process at a time. Projected GPU-minutes are posted before each setting.
Gate at each setting (as 1.1): mamba_ssm selective >= 0.95 on 3/3 AND selective mean -
ablated mean >= 0.5 at the final eval. The Transformer is reported alongside.
Configs: `mamba_ssm_small.yaml` (lr 1e-3), `transformer_small.yaml` (lr 1e-3,
max_seq_len 512: the 1024 settings need an author decision on this before they run).

## Setting 1: seq_len 256, n_data 4: pilots (allowed now by the author)

The ablated arm waits for the 1.1c gate.

**Prediction.** Both pilots reach 0.95 on every seed, later than at seq_len 64 (where
mamba_ssm took 1250-1500 steps): the data are 4x sparser in the input region. I expect
mamba_ssm steps-to-0.95 in roughly 2000-6000 and the Transformer in a similar range; no
confident ordering between them.

**Budget.** No figure was set for the pilots; I cap this kernel at 40 GPU-min. Expected:
~4 min setup plus 6 runs of ~3000-7000 steps at an estimated ~20-30 steps/s at seq_len 256
(not yet measured; the kernel records wall-clock per run), i.e. ~15-30 min. Runs are
ordered mamba s0, Transformer s0, mamba s1, ... so a deadline cut leaves both arms with
the same number of seeds.
