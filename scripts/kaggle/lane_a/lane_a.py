# Lane A, 2026-11-19 (standing approval D6): Phase 2.1 cost sweep, the W1/W2 figure
# models, and the 1.1d rerun on the 20k schedule. Predictions: results/phase2,
# results/figure_models, results/phase1_1d. Push with: kaggle kernels push -p scripts/kaggle/lane_a
import os, subprocess, time

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
W = "/kaggle/working/out"
T0 = time.time()


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
    print(f"[{(time.time()-T0)/60:5.1f} min] {msg}", flush=True)


def train(outdir, name, cmd, deadline):
    os.makedirs(f"{outdir}/logs", exist_ok=True)
    left = deadline - time.time()
    if left <= 0:
        stamp(f"{name}: not_run (deadline)"); return "not_run"
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

stamp("2.1 cost sweep")
os.makedirs(f"{W}/phase2", exist_ok=True)
rc = run(f"python scripts/cost_scaling.py --out {W}/phase2/cost_scaling.json", check=False,
         timeout=22 * 60)
status["2.1"] = "done" if rc == 0 else f"rc={rc}"

stamp("figure models")
TA = "--task selective_copy --task_args n_data=4,seq_len=64,vocab_size=64"
COMMON = (f"{TA} --steps 20000 --schedule_steps 20000 --early_stop_acc 0.99 "
          "--early_stop_evals 3 --seed 0 --device cuda")
fd = time.time() + 10 * 60
for name, cfg in (("transformer_s0", "transformer_small"), ("mamba_ssm_s0", "mamba_ssm_small")):
    status[f"fig_{name}"] = train(f"{W}/figure_models", name,
        f"python src/train_task.py --config configs/recall/{cfg}.yaml {COMMON} "
        f"--save_model {W}/figure_models/{name}.pt", fd)

stamp("1.1d rerun")
dd = time.time() + 24 * 60
SSM = ("python src/train_task.py --config configs/recall/mamba_ssm_small.yaml "
       "--task selective_copy --steps 4000 --schedule_steps 20000 --early_stop_acc 0.99 "
       "--early_stop_evals 3 --device cuda --dump_slot_acc")
for seed in (0, 1, 2):
    for nd, arm, flag in ((1, "selective", ""), (2, "selective", ""), (3, "selective", ""),
                          (4, "selective", ""), (1, "ablated", "--ablate"), (4, "ablated", "--ablate")):
        status[f"d_{arm}_nd{nd}_s{seed}"] = train(f"{W}/phase1_1d/sched20k", f"{arm}_nd{nd}_s{seed}",
            f"{SSM} --task_args n_data={nd},seq_len=64,vocab_size=64 --seed {seed} {flag}", dd)
bad = {k: v for k, v in status.items() if v != "done"}
print("LANE_A COMPLETE" if not bad else f"LANE_A INCOMPLETE: {bad}")
