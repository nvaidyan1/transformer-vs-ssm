# Lane C, 2026-11-19 (standing approval D6): plan2.md 5.2 filtering models and the 5.3 gain
# analysis. Prediction: results/phase5/PREDICTION.md.
# Push with: kaggle kernels push -p scripts/kaggle/lane_c
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/out/phase5"
TRAIN_DEADLINE_H = 4.5
T0 = time.time()


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")
    return r.returncode


def stamp(msg):
    print(f"[{(time.time()-T0)/60:6.1f} min] {msg}", flush=True)


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
os.makedirs(f"{OUT}/models", exist_ok=True)
deadline = time.time() + TRAIN_DEADLINE_H * 3600
bad = {}
for L in (1, 2):                                  # the gates are at 1 layer: run those first
    for seed in (0, 1, 2):
        for task in ("switching", "stationary", "dropout"):
            for arch in ("mamba", "mamba_ablated", "gru"):
                name = f"{task}_{arch}_L{L}_s{seed}"
                left = deadline - time.time()
                if left <= 0:
                    stamp(f"{name}: not_run (deadline)"); bad[name] = "not_run"; continue
                cmd = (f"python src/train_filter.py --task {task} --arch {arch} --n_layers {L} "
                       f"--seed {seed} --steps 3000 --device cuda --max_minutes 5 "
                       f"--out {OUT}/{name}.json --save_model {OUT}/models/{name}.pt")
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
stamp("5.3 gain analysis")
rc = run(f"python scripts/phase5_gain.py --models {OUT}/models --out {OUT}/gain.json --device cuda",
         check=False)
if rc != 0:
    bad["5.3"] = f"rc={rc}"
print("LANE_C COMPLETE" if not bad else f"LANE_C INCOMPLETE: {bad}")
