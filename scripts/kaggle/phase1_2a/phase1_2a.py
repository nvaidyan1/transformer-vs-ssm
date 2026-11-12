# plan2.md 1.2a on a Kaggle T4: official-model ablation tests (CUDA), then selective vs
# ablated mamba_ssm at seq_len 64, seed 0, CAP 3000. Prediction:
# results/phase1_2a/PREDICTION.md. Push with: kaggle kernels push -p scripts/kaggle/phase1_2a
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase1_2a"
TRAIN_DEADLINE_MIN = 15
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
# the 1.2a tests by name first (verbose, so the CUDA variants are visibly run), then all
run("python -m pytest tests/test_mamba_official.py -v -rs -p no:warnings -k ablation")
run("python -m pytest tests/ -q -p no:warnings")
os.makedirs(f"{OUT}/logs", exist_ok=True)
deadline = time.time() + TRAIN_DEADLINE_MIN * 60
BASE = ("python src/train_task.py --config configs/recall/mamba_ssm_small.yaml "
        "--task selective_copy --task_args n_data=4,seq_len=64,vocab_size=64 "
        "--steps 3000 --seed 0 --early_stop_acc 0.99 --early_stop_evals 3 --device cuda")
ok = True
for name, extra in (("official_selective_s0", ""), ("official_ablated_s0", "--ablate")):
    left = deadline - time.time()
    if left <= 0:
        stamp(f"{name}: not_run (deadline)"); ok = False; continue
    with open(f"{OUT}/logs/{name}.log", "w") as log:
        try:
            r = subprocess.run(f"{BASE} {extra} --out {OUT}/{name}.json", shell=True,
                               stdout=log, stderr=subprocess.STDOUT, timeout=left)
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    ok = ok and status == "done"
    stamp(f"{name}: {status}")
print("PHASE1_2A COMPLETE" if ok else "PHASE1_2A INCOMPLETE")
