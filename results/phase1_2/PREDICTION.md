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

### Setting 1 pilots: outcome (appended after the run; prediction above unchanged)

Kaggle T4, commit 4dee336, ~31 GPU-min of the 40 cap. 143 tests passed.
- mamba_ssm selective: steps-to-0.95 1750 / 1750 / 2500; early stop at 3000 / 2750 / 4000;
  final 0.997 / 1.000 / 0.999. ~46 steps/s. **CAP = 4000.**
- Transformer: steps-to-0.95 12750 / 15000 / 14250; early stop only on seed 2 (16750);
  seeds 0 and 1 ran the full 20,000; final 0.980 / 0.955 / 0.993. ~39.5 steps/s. It reached
  ~0.8 within 500 steps on every seed, then rose slowly for >10,000 steps.
- Prediction: both reach 0.95 on every seed (held); "a similar range, no confident
  ordering" (wrong): the Transformer needed 6-8x more steps than mamba_ssm.
- The ablated arm is blocked: the 1.1c gate failed (results/phase1_1c).

### Setting 1 ablated arm (2026-11-12; cleared by the 1.1c post-hoc amendment)

Written before the run. mamba_ssm `--ablate`, seeds 0-2, `--steps 4000` (CAP), early
stop as above, one process at a time. Approved budget 10 GPU-min.
**Prediction:** stays near chance (1/62) on every seed; **gate PASS** (selective
0.997 / 1.000 / 0.999 from the pilots). The pilots ran with a 20,000-step schedule and
the ablated arm runs with a 4000-step one; the gate compares final evals as specified.

### Setting 1 ablated arm: outcome (appended after the run)

Kaggle T4, commit 3242c64, ~7.3 GPU-min incl. setup (of 10). Ablated final 0.017 / 0.012 /
0.019 (chance 0.016). **Setting 1 gate PASS**: selective 3/3 >= 0.95 (0.997 / 1.000 /
0.999); selective mean - ablated mean = +0.983. As predicted.

## Setting 2: seq_len 1024, n_data 4 (approved 2026-11-12; under the scheduling rule)

Written before the run. Every run uses `--schedule_steps 20000`. Kernel 1 (capped at 60
GPU-min): mamba_ssm selective pilot, seeds 0-2, up to 20,000 steps with early stop; then
the Transformer (`transformer_small_seq1024.yaml`: max_seq_len 1024, 505,544 params =
399,048 + 106,496 positional table), seed 0, CAP 20,000, early stop as usual, reference
arm only, not gated. Kernel 2 (cap posted with the pilot results): mamba_ssm ablated,
seeds 0-2 to CAP, plus seed 0 to the full 20,000 steps.

**Prediction.** mamba_ssm selective reaches 0.95 on every seed, later than at 256
(1750-2500): roughly 2500-6000 steps. The Transformer, which needed 12,750-15,000 steps
at seq_len 256, does not reach 0.95 within 20,000 steps at 1024 (low confidence). Gate
(once the ablated arm runs): PASS, with the ablated arm near chance, including the
20,000-step seed.
