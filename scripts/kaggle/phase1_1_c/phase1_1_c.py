# plan2.md 1.1 Stage C on a Kaggle T4: custom selective and custom ablated, seeds 0-2,
# CAP 3000 steps, one process at a time, config as committed (no overrides). If the
# ablated final mean is below 0.5, two more ablated runs at seed 0, lr 3e-4 and 3e-3.
# Prediction and rules: results/phase1_1/PREDICTION.md (amendment before Stage C).
# Push with: kaggle kernels push -p scripts/kaggle/phase1_1_c
import json, os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase1_1/stage_c"
CAP = 3000
TRAIN_DEADLINE_MIN = 66      # Stage C budget 70 GPU-min incl. setup
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
        "--task selective_copy --task_args n_data=4,seq_len=64,vocab_size=64 "
        f"--steps {CAP} --early_stop_acc 0.99 --early_stop_evals 3 --device cuda")


def train(name, extra):
    left = deadline - time.time()
    if left <= 0:
        stamp(f"{name}: not_run (deadline)")
        return "not_run"
    with open(f"{OUT}/logs/{name}.log", "w") as log:
        try:
            r = subprocess.run(f"{BASE} {extra} --out {OUT}/{name}.json", shell=True,
                               stdout=log, stderr=subprocess.STDOUT, timeout=left)
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    stamp(f"{name}: {status}")
    return status


stamp("STAGE C training start")
status = {}
for seed in (0, 1, 2):
    status[f"custom_selective_s{seed}"] = train(f"custom_selective_s{seed}", f"--seed {seed}")
    status[f"custom_ablated_s{seed}"] = train(f"custom_ablated_s{seed}", f"--seed {seed} --ablate")

finals = []
for seed in (0, 1, 2):
    p = f"{OUT}/custom_ablated_s{seed}.json"
    if os.path.exists(p):
        r = json.load(open(p))
        if r.get("complete"):
            finals.append(r["history"][-1]["val_acc"])
abl_mean = sum(finals) / len(finals) if len(finals) == 3 else None
stamp(f"ablated final mean at lr 1e-3: {abl_mean}")
if abl_mean is not None and abl_mean < 0.5:
    for lr in ("3e-4", "3e-3"):
        n = f"custom_ablated_lr{lr}_s0"
        status[n] = train(n, f"--seed 0 --ablate --lr {lr}")
else:
    stamp("extra ablated lr runs not triggered")
json.dump({"status": status, "ablated_mean_lr1e-3": abl_mean,
           "train_wall_min": round((time.time() - T0) / 60, 2)},
          open(f"{OUT}/manifest.json", "w"), indent=2)
print("STAGE C COMPLETE" if all(v == "done" for v in status.values()) else "STAGE C INCOMPLETE")
