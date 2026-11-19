# Lane A2, 2026-11-19 (standing approval D6): plan2.md 4.1 MQAR capacity sweep, CAP 4000
# on the 20k schedule (D3). Prediction: results/phase4/PREDICTION.md.
# Push with: kaggle kernels push -p scripts/kaggle/lane_a2
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/out/phase4"
TRAIN_DEADLINE_H = 3.0
T0 = time.time()


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")


def stamp(msg):
    print(f"[{(time.time()-T0)/60:6.1f} min] {msg}", flush=True)


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
deadline = time.time() + TRAIN_DEADLINE_H * 3600
PAIRS = (4, 64, 16, 8, 32)          # gate-critical pair counts first within each seed
MODELS = ("mamba_ssm_d32", "transformer_m32", "mamba_ssm_d64", "transformer_m64",
          "mamba_ssm_d128", "transformer_m128")
bad = {}
for seed in (0, 1, 2):
    for p in PAIRS:
        for m in MODELS:
            name = f"{m}_p{p}_s{seed}"
            left = deadline - time.time()
            if left <= 0:
                stamp(f"{name}: not_run (deadline)"); bad[name] = "not_run"; continue
            cmd = (f"python src/train_task.py --config configs/mqar/{m}.yaml --task mqar "
                   f"--task_args n_pairs={p},seq_len={4 * p},vocab_size=256 "
                   f"--steps 4000 --schedule_steps 20000 --early_stop_acc 0.99 "
                   f"--early_stop_evals 3 --seed {seed} --device cuda --out {OUT}/{name}.json")
            t = time.time()
            with open(f"{OUT}/logs/{name}.log", "w") as log:
                try:
                    r = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=left)
                    status = "done" if r.returncode == 0 else f"failed({r.returncode})"
                except subprocess.TimeoutExpired:
                    status = "timeout"
            if status != "done":
                bad[name] = status
            stamp(f"{name}: {status} ({time.time() - t:.0f}s)")
print("LANE_A2 COMPLETE" if not bad else f"LANE_A2 INCOMPLETE: {bad}")
