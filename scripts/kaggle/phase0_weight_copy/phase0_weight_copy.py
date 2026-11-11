# plan2.md 0.2 follow-up (amended): init-std check with the real mamba-ssm mixer, then
# the weight-copied custom vs official pair at emb_init_std=1.0, decay_1d=False, seed 0.
# Push with: kaggle kernels push -p scripts/kaggle/phase0_weight_copy
import os, subprocess

COMMIT = "main"
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase0_weight_copy"
PAIR_TIMEOUT_MIN = 13


def run(cmd, check=True, timeout=None):
    print(f"\n$ {cmd}", flush=True)
    try:
        r = subprocess.run(cmd, shell=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout}s: {cmd}", flush=True)
        return None
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED ({r.returncode}): {cmd}")
    return r.returncode


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
os.makedirs(OUT, exist_ok=True)
run(f"python scripts/init_std_compare.py --out {OUT}/init_std_mamba_ssm.json")
rc = run("python scripts/phase0_weight_copy.py --emb_init_std 1.0 --decay_1d false "
         f"--seed 0 --steps 2000 --device cuda --out {OUT}/pair_emb1.0_d1d0_s0.json",
         check=False, timeout=PAIR_TIMEOUT_MIN * 60)
print("PAIR COMPLETE" if rc == 0 else f"PAIR NOT COMPLETE (rc={rc})")
