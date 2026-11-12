# plan2.md 1.1 (amended) Stages A and B on a Kaggle T4. Prediction:
# results/phase1_1/PREDICTION.md. Push with: kaggle kernels push -p scripts/kaggle/phase1_1_ab
#   Stage A (max 10 GPU-min, incl. setup): custom Mamba speed in 4 configurations,
#     CUDA parity of the fastest, parity tests, throughput with 1/2/3 processes.
#   Stage B (max 20 GPU-min, incl. the mamba-ssm install): mamba_ssm pilot, 3 seeds,
#     up to 20,000 steps, early stop at val_acc > 0.99 for 3 evals in a row.
import glob, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase1_1"
A_TRAIN_MIN = 7.5          # after ~2 min setup
B_TRAIN_MIN = 16.0         # after ~2.5 min mamba-ssm install
T_KERNEL0 = time.time()


def run(cmd, check=True, timeout=None):
    print(f"\n$ {cmd}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {cmd}", flush=True)
        return None
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")
    return r.returncode


def stamp(msg):
    print(f"[{(time.time()-T_KERNEL0)/60:5.1f} min] {msg}", flush=True)


os.chdir("/kaggle/working")
if not os.path.exists("transformer-vs-ssm"):
    run(f"git clone {REPO}")
os.chdir("transformer-vs-ssm")
run(f"git checkout {COMMIT}")
run("git log --oneline -1")
run("nvidia-smi --query-gpu=name --format=csv,noheader")
run("pip install -q pyyaml pytest einops")
run("python -m pytest tests/ -q -p no:warnings")
A, B = f"{OUT}/stage_a", f"{OUT}/stage_b"
os.makedirs(A, exist_ok=True)
os.makedirs(f"{B}/logs", exist_ok=True)

# ---------------- Stage A ----------------
stamp("STAGE A start")
a_deadline = time.time() + A_TRAIN_MIN * 60
BENCH = "python scripts/bench_custom_mamba.py --steps 200 --warmup 20 --device cuda"
for scan in ("sequential", "parallel"):
    for ck in ("true", "false"):
        left = a_deadline - time.time()
        if left > 0:
            run(f"{BENCH} --scan {scan} --grad_checkpoint {ck} "
                f"--out {A}/config_{scan}_ckpt{ck}.json", check=False, timeout=left)
cfgs = [json.load(open(p)) for p in glob.glob(f"{A}/config_*.json")]
fast = max(cfgs, key=lambda r: r["steps_per_s"])
fs, fc = fast["scan"], str(fast["grad_checkpoint"]).lower()
stamp(f"fastest: scan={fs} grad_checkpoint={fc} ({fast['steps_per_s']:.2f} steps/s)")
run(f"python scripts/bench_custom_mamba.py --scan {fs} --grad_checkpoint {fc} --parity "
    f"--device cuda --out {A}/parity_{fs}_ckpt{fc}.json", check=False,
    timeout=max(1, a_deadline - time.time()))
run("python -m pytest tests/test_mamba_scan.py tests/test_selective_scan_parity.py "
    "tests/test_mamba_ablation.py -q -p no:warnings", check=False)
for k in (1, 2, 3):
    left = a_deadline - time.time()
    if left < 60:
        stamp(f"skip concurrency k={k}: Stage A deadline")
        continue
    start_at = time.time() + 45
    procs = [subprocess.Popen(f"{BENCH} --scan {fs} --grad_checkpoint {fc} "
                              f"--start_at {start_at} --tag k{k}p{i} "
                              f"--out {A}/conc_k{k}_p{i}.json", shell=True)
             for i in range(k)]
    for p in procs:
        try:
            p.wait(timeout=max(1, a_deadline - time.time()))
        except subprocess.TimeoutExpired:
            p.kill()
    stamp(f"concurrency k={k} done")
stamp("STAGE A end")

# ---------------- Stage B ----------------
stamp("STAGE B start (install)")
run("pip install -q packaging ninja wheel psutil")
run("pip install -q 'causal-conv1d>=1.4.0' --no-build-isolation", check=False)
run("pip install -q mamba-ssm --no-build-isolation")
b_deadline = time.time() + B_TRAIN_MIN * 60
stamp("STAGE B training start")


def pilot(seed):
    name = f"mamba_ssm_s{seed}"
    with open(f"{B}/logs/{name}.log", "w") as log:
        try:
            r = subprocess.run(
                "python src/train_task.py --config configs/recall/mamba_ssm_small.yaml "
                "--task selective_copy --task_args n_data=4,seq_len=64,vocab_size=64 "
                f"--steps 20000 --seed {seed} --early_stop_acc 0.99 --early_stop_evals 3 "
                f"--device cuda --out {B}/{name}.json", shell=True, stdout=log,
                stderr=subprocess.STDOUT, timeout=max(1, b_deadline - time.time()))
            status = "done" if r.returncode == 0 else f"failed({r.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    stamp(f"{name}: {status}")
    return status


with ThreadPoolExecutor(3) as ex:
    statuses = list(ex.map(pilot, (0, 1, 2)))
stamp("STAGE B end")
print("PHASE1_1_AB COMPLETE" if all(s == "done" for s in statuses) else "PHASE1_1_AB INCOMPLETE")
