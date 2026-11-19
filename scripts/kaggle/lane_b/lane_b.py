# Lane B, 2026-11-19 (standing approval D6): plan2.md 3.2 TCN reach curve, scope D2,
# CAP 4000 on the 20k schedule (D3). Prediction: results/phase3/PREDICTION.md.
# Push with: kaggle kernels push -p scripts/kaggle/lane_b
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/out/phase3"
TRAIN_DEADLINE_H = 3.5
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
GAPS = (32, 160, 256, 512, 64, 96)          # gate-critical gaps first within each seed
JOBS = []
for seed in (0, 1, 2):
    for g in GAPS:
        for arm in ("tcn_short", "tcn_small"):
            JOBS.append((arm, f"configs/recall/{arm}.yaml", g, seed))
    for g in (64, 256):
        JOBS.append(("transformer", "configs/recall/transformer_small.yaml", g, seed))
        JOBS.append(("mamba_ssm", "configs/recall/mamba_ssm_small.yaml", g, seed))
bad = {}
for arm, cfg, g, seed in JOBS:
    name = f"{arm}_g{g}_s{seed}"
    left = deadline - time.time()
    if left <= 0:
        stamp(f"{name}: not_run (deadline)"); bad[name] = "not_run"; continue
    cmd = (f"python src/train_task.py --config {cfg} --task selective_copy "
           f"--task_args n_data=4,seq_len={g + 64},vocab_size=64,min_gap={g} "
           f"--steps 4000 --schedule_steps 20000 --early_stop_acc 0.99 --early_stop_evals 3 "
           f"--seed {seed} --device cuda --out {OUT}/{name}.json")
    t = time.time()
    with open(f"{OUT}/logs/{name}.log", "w") as log:
        try:
            r = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, timeout=left)
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    if status != "done":
        bad[name] = status
    stamp(f"{name}: {status} ({time.time() - t:.0f}s)")
print("LANE_B COMPLETE" if not bad else f"LANE_B INCOMPLETE: {bad}")
