# Author's decisions of 2026-11-12, items 6, 4 and 5, in one kernel (one mamba-ssm
# install), each with its own deadline inside its approved budget:
#   item 6: 1.2 seq_len 256 ablated arm, 3 seeds, CAP 4000           (10 GPU-min, incl. setup)
#   item 4: 1.1c run 2 again, CAP 20,000, report only               (6 GPU-min)
#   item 5: 1.1d items the ablated model can carry, 24 runs         (20 GPU-min)
# Predictions: results/phase1_2/PREDICTION.md, results/phase1_1c/PREDICTION.md,
# results/phase1_1d/PREDICTION.md. Push with:
#   kaggle kernels push -p scripts/kaggle/phase1_items456
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
SSM = ("python src/train_task.py --config configs/recall/mamba_ssm_small.yaml "
       "--task selective_copy --early_stop_acc 0.99 --early_stop_evals 3 --device cuda")
status = {}

# item 6: setup counts against its 10 minutes
stamp("ITEM 6 start")
d6 = T0 + 9.5 * 60
for seed in (0, 1, 2):
    status[f"i6_s{seed}"] = train(f"{W}/phase1_2/seq256_nd4", f"mamba_ssm_ablated_s{seed}",
        f"{SSM} --task_args n_data=4,seq_len=256,vocab_size=64 --steps 4000 --seed {seed} --ablate",
        d6)

stamp("ITEM 4 start")
d4 = time.time() + 5.8 * 60
status["i4"] = train(f"{W}/phase1_1c", "run2_fixed_nd1_cap20000",
    f"{SSM} --task_args n_data=1,seq_len=64,vocab_size=64,fixed_positions=1 --steps 20000 "
    f"--seed 0 --ablate --dump_slot_acc", d4)

stamp("ITEM 5 start")
d5 = time.time() + 19.5 * 60
for seed in (0, 1, 2):
    for nd in (1, 2, 3, 4):
        for arm, flag in (("selective", ""), ("ablated", "--ablate")):
            status[f"i5_{arm}_nd{nd}_s{seed}"] = train(f"{W}/phase1_1d", f"{arm}_nd{nd}_s{seed}",
                f"{SSM} --task_args n_data={nd},seq_len=64,vocab_size=64 --steps 4000 "
                f"--seed {seed} {flag} --dump_slot_acc", d5)
stamp("ALL END")
bad = {k: v for k, v in status.items() if v != "done"}
print("ITEMS456 COMPLETE" if not bad else f"ITEMS456 INCOMPLETE: {bad}")
