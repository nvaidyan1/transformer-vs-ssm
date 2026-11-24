# Lane E, 2026-11-24 (author approval): Phase 4 per-slot diagnosis of the 4-pair plateaus.
# Prediction: results/phase4/PREDICTION.md. Push with: kaggle kernels push -p scripts/kaggle/lane_e
import os, subprocess, time

REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/out/phase4/diag"
T0 = time.time()


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")
    return r.returncode


os.chdir("/kaggle/working")
if not os.path.exists("transformer-vs-ssm"):
    run(f"git clone {REPO}")
os.chdir("transformer-vs-ssm")
run("git log --oneline -1")
run("pip install -q packaging ninja wheel psutil pyyaml pytest einops")
run("pip install -q 'causal-conv1d>=1.4.0' --no-build-isolation", check=False)
run("pip install -q mamba-ssm --no-build-isolation")
run("python -m pytest tests/ -q -p no:warnings")
os.makedirs(f"{OUT}/logs", exist_ok=True)
bad = []
for seed in (0, 1, 2):
    for m in ("transformer_m32", "mamba_ssm_d32"):
        name = f"{m}_p4_s{seed}"
        rc = run(f"python src/train_task.py --config configs/mqar/{m}.yaml --task mqar "
                 f"--task_args n_pairs=4,seq_len=16,vocab_size=256 --steps 4000 "
                 f"--schedule_steps 20000 --early_stop_acc 0.99 --early_stop_evals 3 "
                 f"--seed {seed} --device cuda --dump_slot_acc --dump_answer_hist "
                 f"--save_model {OUT}/{name}.pt --out {OUT}/{name}.json "
                 f"> {OUT}/logs/{name}.log 2>&1", check=False)
        print(f"[{(time.time()-T0)/60:5.1f} min] {name}: rc={rc}", flush=True)
        if rc:
            bad.append(name)
print("LANE_E COMPLETE" if not bad else f"LANE_E INCOMPLETE: {bad}")
