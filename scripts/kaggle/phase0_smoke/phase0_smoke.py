# plan2.md 0.1 gate on a Kaggle T4: full test suite WITH mamba-ssm installed (so the
# official-mixer test variants run, not just the vendored stand-in), then one
# 50-step smoke run per arch through src/train_task.py.
# Push with: kaggle kernels push -p scripts/kaggle/phase0_smoke
# Set COMMIT to the commit under test before pushing.
import os, subprocess, sys

COMMIT = os.environ.get("COMMIT", "main")
REPO = "https://github.com/nvaidyan1/transformer-vs-ssm.git"
OUT = "/kaggle/working/phase0_smoke"


def run(cmd, check=True):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
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
run("pip install causal-conv1d>=1.4.0 --no-build-isolation", check=False)
run("pip install mamba-ssm --no-build-isolation")
run("python -c \"import mamba_ssm, torch; print('mamba_ssm', mamba_ssm.__version__, "
    "'torch', torch.__version__)\"")

# -rs lists skips, so we can see the official-mixer variants actually ran.
run("python -m pytest tests/ -q -rs 2>&1 | tail -15", check=False)
run("python -m pytest tests/ -q -p no:warnings")

os.makedirs(OUT, exist_ok=True)
common = ("--task selective_copy --task_args n_data=4,seq_len=64 --steps 50 "
          "--device cuda")
for name, args in [("transformer", "--config configs/recall/transformer_small.yaml"),
                   ("tcn", "--config configs/recall/tcn_small.yaml"),
                   ("mamba", "--config configs/recall/mamba_small.yaml"),
                   ("mamba_ssm", "--config configs/recall/mamba_small.yaml --arch mamba_ssm")]:
    run(f"python src/train_task.py {args} {common} --out {OUT}/{name}.json")
print("\nPHASE0 SMOKE OK")
