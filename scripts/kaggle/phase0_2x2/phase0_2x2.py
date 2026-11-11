# plan2.md 0.2 on a Kaggle T4: custom vs official Mamba, emb_init_std x decay_1d,
# 3 seeds each. See results/phase0_2x2/PREDICTION.md for the prediction and gate.
# Push with: kaggle kernels push -p scripts/kaggle/phase0_2x2
#
# Budget guard: all training must finish within TRAIN_DEADLINE_MIN of the first job
# starting. Jobs still running at the deadline are killed and recorded as
# "timeout"; jobs not started are recorded as "not_run". No retries, no extension.
import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase0_2x2"
TRAIN_DEADLINE_MIN = 50
WORKERS = 4
SEEDS = (0, 1, 2)
EMB = (0.02, 1.0)
DECAY_1D = (False, True)
COMMON = ("--config configs/recall/mamba_small.yaml --task selective_copy "
          "--task_args n_data=4,seq_len=64,vocab_size=64 --steps 2000 --batch_size 32 "
          "--lr 3e-4 --device cuda --quiet")


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

# Official first (fast), then custom; seed-major within each so a deadline cut
# leaves every cell with the same number of seeds.
jobs = []
for arch in ("mamba_ssm", "mamba"):
    for seed in SEEDS:
        for emb in EMB:
            for d1d in DECAY_1D:
                name = f"{arch}_emb{emb}_d1d{int(d1d)}_s{seed}"
                cmd = (f"python src/train_task.py {COMMON} --arch {arch} --seed {seed} "
                       f"--emb_init_std {emb} --out_proj_depth_scale true "
                       f"--decay_1d {str(d1d).lower()} --out {OUT}/{name}.json")
                jobs.append(dict(name=name, arch=arch, seed=seed, emb_init_std=emb,
                                 decay_1d=d1d, cmd=cmd))

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
    print(f"[{(time.time()-t_start)/60:5.1f} min] {job['name']:32s} {status:10s} "
          f"{res['wall_s']:7.1f}s", flush=True)
    return res


with ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(work, jobs))

manifest = {"commit": commit, "workers": WORKERS,
            "train_deadline_min": TRAIN_DEADLINE_MIN,
            "train_wall_min": round((time.time() - t_start) / 60, 2),
            "jobs": results}
json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=2)
n_done = sum(r["status"] == "done" for r in results)
print(f"\n{n_done}/{len(results)} jobs done in {manifest['train_wall_min']} min "
      f"(commit {commit})")
print("PHASE0_2X2 COMPLETE" if n_done == len(results) else "PHASE0_2X2 INCOMPLETE")
