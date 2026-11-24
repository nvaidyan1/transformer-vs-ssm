# Lane D, 2026-11-23 (author approval): 5.3 post-hoc analysis, P3a (seed 2 for the four
# tcn_short gate cells), P4a (D3 extensions for the six 4-pair gate cells).
# Predictions: results/phase5, results/phase3, results/phase4 PREDICTION.md.
# Push with: kaggle kernels push -p scripts/kaggle/lane_d
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
W = "/kaggle/working/out"
TRAIN_DEADLINE_H = 2.0
T0 = time.time()


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")
    return r.returncode


def stamp(msg):
    print(f"[{(time.time()-T0)/60:6.1f} min] {msg}", flush=True)


def train(outdir, name, cmd, deadline):
    os.makedirs(f"{outdir}/logs", exist_ok=True)
    left = deadline - time.time()
    if left <= 0:
        stamp(f"{name}: not_run (deadline)"); return "not_run"
    t = time.time()
    with open(f"{outdir}/logs/{name}.log", "w") as log:
        try:
            r = subprocess.run(f"{cmd} --out {outdir}/{name}.json", shell=True, stdout=log,
                               stderr=subprocess.STDOUT, timeout=left)
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    stamp(f"{name}: {status} ({time.time() - t:.0f}s)")
    return status


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
deadline = time.time() + TRAIN_DEADLINE_H * 3600
status = {}

stamp("5.3 post-hoc gain analysis")
os.makedirs(f"{W}/phase5", exist_ok=True)
rc = run(f"python scripts/phase5_gain.py --models results/phase5/models "
         f"--out {W}/phase5/gain.json --device cuda", check=False)
status["5.3"] = "done" if rc == 0 else f"rc={rc}"

stamp("P3a")
for g in (32, 160, 256, 512):
    status[f"p3_g{g}"] = train(f"{W}/phase3", f"tcn_short_g{g}_s2",
        f"python src/train_task.py --config configs/recall/tcn_short.yaml --task selective_copy "
        f"--task_args n_data=4,seq_len={g + 64},vocab_size=64,min_gap={g} --steps 4000 "
        f"--schedule_steps 20000 --early_stop_acc 0.99 --early_stop_evals 3 --seed 2 "
        f"--device cuda", deadline)

stamp("P4a")
for seed in (0, 1, 2):
    for m in ("mamba_ssm_d32", "transformer_m32"):
        status[f"p4_{m}_s{seed}"] = train(f"{W}/phase4/ext", f"{m}_p4_s{seed}",
            f"python src/train_task.py --config configs/mqar/{m}.yaml --task mqar "
            f"--task_args n_pairs=4,seq_len=16,vocab_size=256 --steps 20000 "
            f"--schedule_steps 20000 --early_stop_acc 0.99 --early_stop_evals 3 "
            f"--seed {seed} --device cuda", deadline)
bad = {k: v for k, v in status.items() if v != "done"}
print("LANE_D COMPLETE" if not bad else f"LANE_D INCOMPLETE: {bad}")
