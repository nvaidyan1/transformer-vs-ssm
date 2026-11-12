# plan2.md 1.2, setting seq_len 256 / n_data 4: mamba_ssm selective and Transformer
# pilots, 3 seeds each, up to 20,000 steps with early stop. One process at a time.
# Prediction: results/phase1_2/PREDICTION.md. Push with:
#   kaggle kernels push -p scripts/kaggle/phase1_2_pilot256
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase1_2/seq256_nd4"
TRAIN_DEADLINE_MIN = 35          # 40 GPU-min cap incl. ~4 min setup
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
COMMON = ("--task selective_copy --task_args n_data=4,seq_len=256,vocab_size=64 "
          "--steps 20000 --early_stop_acc 0.99 --early_stop_evals 3 --device cuda")
ARMS = (("mamba_ssm_selective", "configs/recall/mamba_ssm_small.yaml"),
        ("transformer", "configs/recall/transformer_small.yaml"))
ok = True
for seed in (0, 1, 2):
    for arm, cfg in ARMS:
        name = f"{arm}_s{seed}"
        left = deadline - time.time()
        if left <= 0:
            stamp(f"{name}: not_run (deadline)"); ok = False; continue
        with open(f"{OUT}/logs/{name}.log", "w") as log:
            try:
                r = subprocess.run(f"python src/train_task.py --config {cfg} {COMMON} "
                                   f"--seed {seed} --out {OUT}/{name}.json", shell=True,
                                   stdout=log, stderr=subprocess.STDOUT, timeout=left)
                status = "done" if r.returncode == 0 else f"failed({r.returncode})"
            except subprocess.TimeoutExpired:
                status = "timeout"
        ok = ok and status == "done"
        stamp(f"{name}: {status}")
print("PILOT256 COMPLETE" if ok else "PILOT256 INCOMPLETE")
