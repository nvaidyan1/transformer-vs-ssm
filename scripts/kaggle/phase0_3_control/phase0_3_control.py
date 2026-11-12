# plan2.md 0.3 Transformer control: 3 seeds, 4000 steps, lr from the committed config
# (no --lr override). Prediction and gate: results/phase0_3_lr/PREDICTION.md, Part B.
# Push with: kaggle kernels push -p scripts/kaggle/phase0_3_control
# No mamba-ssm needed; the suite's mamba-ssm tests skip.
import os, subprocess
from concurrent.futures import ThreadPoolExecutor

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase0_3_control"
TIMEOUT_MIN = 12


def run(cmd, check=True, **kw):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")
    return r


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


def train(seed):
    name = f"transformer_s{seed}"
    with open(f"{OUT}/logs/{name}.log", "w") as log:
        try:
            r = subprocess.run(
                "python src/train_task.py --config configs/recall/transformer_small.yaml "
                "--task selective_copy --task_args n_data=4,seq_len=64,vocab_size=64 "
                f"--steps 4000 --seed {seed} --device cuda --out {OUT}/{name}.json",
                shell=True, stdout=log, stderr=subprocess.STDOUT, timeout=TIMEOUT_MIN * 60)
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    print(f"{name}: {status}", flush=True)
    return status


with ThreadPoolExecutor(3) as ex:
    statuses = list(ex.map(train, (0, 1, 2)))
print("CONTROL COMPLETE" if all(s == "done" for s in statuses) else "CONTROL INCOMPLETE")
