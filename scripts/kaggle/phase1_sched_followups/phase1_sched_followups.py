# Schedule follow-ups (author, 2026-11-12), items b, d, c. 30 GPU-min, report only.
# Prediction: results/phase1_sched_followups/PREDICTION.md. Push with:
#   kaggle kernels push -p scripts/kaggle/phase1_sched_followups
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
ABL = ("python src/train_task.py --config configs/recall/mamba_ssm_small.yaml "
       "--task selective_copy --schedule_steps 20000 --device cuda --ablate")
ES = "--early_stop_acc 0.99 --early_stop_evals 3"
stamp("(b) start")
for seed in (0, 1, 2):
    status[f"b_s{seed}"] = train(f"{W}/phase1_2/seq256_nd4_sched20k", f"mamba_ssm_ablated_s{seed}",
        f"{ABL} {ES} --task_args n_data=4,seq_len=256,vocab_size=64 --steps 4000 --seed {seed}",
        T0 + 10 * 60)
stamp("(d) start")
for seed in (0, 1, 2):
    for nd in (2, 3):
        status[f"d_nd{nd}_s{seed}"] = train(f"{W}/phase1_1d/sched20k", f"ablated_nd{nd}_s{seed}",
            f"{ABL} {ES} --task_args n_data={nd},seq_len=64,vocab_size=64 --steps 4000 "
            f"--seed {seed} --dump_slot_acc", T0 + 16.5 * 60)
stamp("(c) start")
for L in (64, 256):
    status[f"c_seq{L}"] = train(f"{W}/phase1_sched_followups/longbudget", f"ablated_nd4_seq{L}_s0",
        f"{ABL} --task_args n_data=4,seq_len={L},vocab_size=64 --steps 20000 --seed 0 "
        f"--dump_slot_acc", T0 + 29 * 60)
bad = {k: v for k, v in status.items() if v != "done"}
print("FOLLOWUPS COMPLETE" if not bad else f"FOLLOWUPS INCOMPLETE: {bad}")
