# plan2.md 1.1c on a Kaggle T4: what can the ablated model learn? Gate run first.
# Prediction: results/phase1_1c/PREDICTION.md. Push with:
#   kaggle kernels push -p scripts/kaggle/phase1_1c
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase1_1c"
TRAIN_DEADLINE_MIN = 25
T0 = time.time()


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")


def stamp(msg):
    print(f"[{(time.time()-T0)/60:5.1f} min] {msg}", flush=True)


os.chdir("/kaggle/working")
if not os.path.exists("transformer-vs-ssm"):
    run(f"git clone {REPO}")
os.chdir("transformer-vs-ssm")
run(f"git checkout {COMMIT}")
run("git log --oneline -1")
run("nvidia-smi --query-gpu=name --format=csv,noheader")
run("pip install -q packaging ninja wheel psutil pyyaml pytest einops")
run("pip install -q 'causal-conv1d>=1.4.0' --no-build-isolation", check=False)
run("pip install -q mamba-ssm --no-build-isolation")
run("python -m pytest tests/ -q -p no:warnings")
os.makedirs(f"{OUT}/logs", exist_ok=True)
deadline = time.time() + TRAIN_DEADLINE_MIN * 60
COMMON = ("--task selective_copy --steps 3000 --seed 0 --early_stop_acc 0.99 "
          "--early_stop_evals 3 --device cuda --ablate --dump_slot_acc")
SSM = f"python src/train_task.py --config configs/recall/mamba_ssm_small.yaml {COMMON}"
CUSTOM = f"python src/train_task.py --config configs/recall/mamba_small.yaml {COMMON}"
TA = "--task_args seq_len=64,vocab_size=64"
JOBS = [
    ("run2_fixed_nd1", f"{SSM} {TA},n_data=1,fixed_positions=1"),          # gate
    ("run3_random_nd1", f"{SSM} {TA},n_data=1"),
    ("run4_fixed_nd2", f"{SSM} {TA},n_data=2,fixed_positions=1"),
    # 1.1b fixed_ablated_s0, identical command plus --dump_slot_acc (already in COMMON)
    ("run1_custom_fixed_nd4", f"{CUSTOM} {TA},n_data=4,fixed_positions=1"),
]
ok = True
for name, cmd in JOBS:
    left = deadline - time.time()
    if left <= 0:
        stamp(f"{name}: not_run (deadline)"); ok = False; continue
    with open(f"{OUT}/logs/{name}.log", "w") as log:
        try:
            r = subprocess.run(f"{cmd} --out {OUT}/{name}.json", shell=True, stdout=log,
                               stderr=subprocess.STDOUT, timeout=left)
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    ok = ok and status == "done"
    stamp(f"{name}: {status}")
print("PHASE1_1C COMPLETE" if ok else "PHASE1_1C INCOMPLETE")
