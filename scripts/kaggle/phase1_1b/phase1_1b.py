# plan2.md 1.1b on a Kaggle T4: positive control for the ablation (fixed-position
# Selective Copying), plus a re-run of 1.1's custom_ablated_s0 with answer histograms.
# Prediction: results/phase1_1b/PREDICTION.md. Push with:
#   kaggle kernels push -p scripts/kaggle/phase1_1b
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase1_1b"
TRAIN_DEADLINE_MIN = 23.5
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
run("pip install -q pyyaml pytest einops")
run("python -m pytest tests/ -q -p no:warnings")
os.makedirs(f"{OUT}/logs", exist_ok=True)
deadline = time.time() + TRAIN_DEADLINE_MIN * 60
BASE = ("python src/train_task.py --config configs/recall/mamba_small.yaml "
        "--task selective_copy --steps 3000 --seed 0 --early_stop_acc 0.99 "
        "--early_stop_evals 3 --device cuda")
JOBS = [
    ("fixed_ablated_s0", "--task_args n_data=4,seq_len=64,vocab_size=64,fixed_positions=1 --ablate"),
    ("fixed_selective_s0", "--task_args n_data=4,seq_len=64,vocab_size=64,fixed_positions=1"),
    # identical to 1.1 Stage C custom_ablated_s0, plus the histogram dump
    ("rerun_custom_ablated_s0", "--task_args n_data=4,seq_len=64,vocab_size=64 --ablate "
                                "--dump_answer_hist"),
]
stamp("training start")
ok = True
for name, extra in JOBS:
    left = deadline - time.time()
    if left <= 0:
        stamp(f"{name}: not_run (deadline)")
        ok = False
        continue
    with open(f"{OUT}/logs/{name}.log", "w") as log:
        try:
            r = subprocess.run(f"{BASE} {extra} --out {OUT}/{name}.json", shell=True,
                               stdout=log, stderr=subprocess.STDOUT, timeout=left)
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    ok = ok and status == "done"
    stamp(f"{name}: {status}")
print("PHASE1_1B COMPLETE" if ok else "PHASE1_1B INCOMPLETE")
