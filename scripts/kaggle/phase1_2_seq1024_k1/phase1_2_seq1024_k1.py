# plan2.md 1.2 setting seq_len 1024 / n_data 4, kernel 1 (capped at 60 GPU-min), under the
# 20,000-step scheduling rule: mamba_ssm selective pilot (3 seeds, early stop), then the
# Transformer reference arm (seed 0, CAP 20,000). Prediction: results/phase1_2/PREDICTION.md.
# Push with: kaggle kernels push -p scripts/kaggle/phase1_2_seq1024_k1
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
W = "/kaggle/working/out"
T0 = time.time()


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")


def stamp(msg):
    print(f"[{(time.time()-T0)/60:5.1f} min] {msg}", flush=True)


def train(outdir, name, cmd, deadline):
    os.makedirs(f"{outdir}/logs", exist_ok=True)
    left = deadline - time.time()
    if left <= 0:
        stamp(f"{name}: not_run (deadline)")
        return "not_run"
    with open(f"{outdir}/logs/{name}.log", "w") as log:
        try:
            r = subprocess.run(f"{cmd} --out {outdir}/{name}.json", shell=True, stdout=log,
                               stderr=subprocess.STDOUT, timeout=left)
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    stamp(f"{name}: {status}")
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
status = {}
COMMON = ("--task selective_copy --task_args n_data=4,seq_len=1024,vocab_size=64 "
          "--steps 20000 --schedule_steps 20000 --early_stop_acc 0.99 --early_stop_evals 3 "
          "--device cuda")
OUT = f"{W}/phase1_2/seq1024_nd4"
deadline = T0 + 58 * 60
for seed in (0, 1, 2):
    status[f"sel_s{seed}"] = train(OUT, f"mamba_ssm_selective_s{seed}",
        f"python src/train_task.py --config configs/recall/mamba_ssm_small.yaml {COMMON} "
        f"--seed {seed}", deadline)
status["tf_s0"] = train(OUT, "transformer_s0",
    f"python src/train_task.py --config configs/recall/transformer_small_seq1024.yaml {COMMON} "
    f"--seed 0", deadline)
bad = {k: v for k, v in status.items() if v != "done"}
print("SEQ1024_K1 COMPLETE" if not bad else f"SEQ1024_K1 INCOMPLETE: {bad}")
