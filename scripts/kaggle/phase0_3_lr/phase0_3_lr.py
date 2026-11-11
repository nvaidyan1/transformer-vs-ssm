# plan2.md 0.3 learning-rate check on a Kaggle T4: lr in {3e-4, 1e-3, 3e-3}, seed 0,
# 4000 steps, Transformer / TCN / mamba_ssm. See results/phase0_3_lr/PREDICTION.md.
# Push with: kaggle kernels push -p scripts/kaggle/phase0_3_lr
#
# Budget guard: 30 GPU-min total; training must finish within TRAIN_DEADLINE_MIN of
# the first job starting. Cut jobs keep their partial JSON (complete=false).
import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase0_3_lr"
TRAIN_DEADLINE_MIN = 24
WORKERS = 4
LRS = ("3e-4", "1e-3", "3e-3")
# slowest first, so the long TCN runs start immediately
ARCHS = (("tcn", "configs/recall/tcn_small.yaml"),
         ("mamba_ssm", "configs/recall/mamba_ssm_small.yaml"),
         ("transformer", "configs/recall/transformer_small.yaml"))
COMMON = ("--task selective_copy --task_args n_data=4,seq_len=64,vocab_size=64 "
          "--steps 4000 --seed 0 --device cuda")


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")


os.chdir("/kaggle/working")
if not os.path.exists("transformer-vs-ssm"):
    run(f"git clone {REPO}")
os.chdir("transformer-vs-ssm")
run(f"git checkout {COMMIT}")
commit = subprocess.run("git rev-parse --short HEAD", shell=True, capture_output=True,
                        text=True).stdout.strip()
run("nvidia-smi --query-gpu=name --format=csv,noheader")
run("pip install -q packaging ninja wheel psutil pyyaml pytest einops")
run("pip install -q 'causal-conv1d>=1.4.0' --no-build-isolation", check=False)
run("pip install -q mamba-ssm --no-build-isolation")
run("python -m pytest tests/ -q -p no:warnings")
os.makedirs(f"{OUT}/logs", exist_ok=True)

jobs = []
for arch, cfg in ARCHS:
    for lr in LRS:
        name = f"{arch}_lr{lr}_s0"
        jobs.append(dict(name=name, arch=arch, lr=lr, cmd=(
            f"python src/train_task.py --config {cfg} {COMMON} --lr {lr} "
            f"--out {OUT}/{name}.json")))

t_start = time.time()
deadline = t_start + TRAIN_DEADLINE_MIN * 60


def work(job):
    remaining = deadline - time.time()
    if remaining <= 0:
        return {**job, "status": "not_run", "wall_s": 0.0}
    t0 = time.time()
    with open(f"{OUT}/logs/{job['name']}.log", "w") as log:
        try:
            r = subprocess.run(job["cmd"], shell=True, stdout=log,
                               stderr=subprocess.STDOUT, timeout=remaining)
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    res = {**job, "status": status, "wall_s": round(time.time() - t0, 1)}
    print(f"[{(time.time()-t_start)/60:5.1f} min] {job['name']:28s} {status:10s} "
          f"{res['wall_s']:7.1f}s", flush=True)
    return res


with ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(work, jobs))
manifest = {"commit": commit, "workers": WORKERS, "train_deadline_min": TRAIN_DEADLINE_MIN,
            "train_wall_min": round((time.time() - t_start) / 60, 2), "jobs": results}
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
n_done = sum(r["status"] == "done" for r in results)
print(f"\n{n_done}/{len(results)} jobs done in {manifest['train_wall_min']} min (commit {commit})")
print("PHASE0_3_LR COMPLETE" if n_done == len(results) else "PHASE0_3_LR INCOMPLETE")
