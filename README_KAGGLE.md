# Running transformer-vs-ssm on Kaggle

Step-by-step instructions to reproduce all results from a fresh Kaggle notebook or GPU instance.  
Expected total runtime: ~6–10 hours on a single T4 GPU (all three models to 50k steps).

---

## Step 1 — Install dependencies

```python
!pip install -q torch numpy pyyaml matplotlib seaborn tqdm
```

Kaggle already ships most of these. The `-q` suppresses output — remove it if you want to verify versions.

---

## Step 2 — Clone the repo

```python
!git clone https://github.com/nvaidyan1/transformer-vs-ssm.git
%cd transformer-vs-ssm
```

---

## Step 3 — Prepare data

Downloads enwik8 (~100MB), unzips it, and saves `data/train.bin` (90M bytes) and `data/val.bin` (~10M bytes).  
Safe to re-run — skips the download if files already exist.

```python
from src.data import prepare_data
prepare_data()
```

Verify:
```python
from src.data import load_split
train = load_split('train')
val   = load_split('val')
print(f'Train: {len(train):,} bytes')   # expect ~90,000,000
print(f'Val:   {len(val):,} bytes')     # expect ~10,000,000
```

---

## Step 4 — Train all three models

Each model trains for 50,000 steps. Checkpoints are saved every 5,000 steps to `checkpoints/{arch}/`.  
If a run is interrupted, re-running the same command resumes from the latest checkpoint automatically.

```bash
# Train sequentially (recommended for a single GPU)
!python src/train.py --config configs/transformer.yaml
!python src/train.py --config configs/tcn.yaml
!python src/train.py --config configs/mamba.yaml
```

**Expected final val_bpc** (approximate, varies slightly by GPU and seed):

| Model       | val_bpc |
|-------------|---------|
| transformer | ~1.3–1.5 |
| tcn         | ~1.4–1.6 |
| mamba       | ~1.3–1.5 |

Log format printed every 100 steps:
```
step   5000 | lr 2.85e-04 | train_loss 1.05 | train_bpc 1.51 | val_bpc 1.48 | 42.3s
```

To monitor a run from a separate cell without blocking:
```python
import subprocess
proc = subprocess.Popen(['python', 'src/train.py', '--config', 'configs/mamba.yaml'])
# proc.wait() to block, or let it run in background
```

---

## Step 5 — Run the length sweep

Measures forward-pass time and peak GPU memory at sequence lengths [128, 256, 512, 1024, 2048, 4096] for all three models.  
Saves results to `checkpoints/sweep_results.json`.

```bash
!bash scripts/sweep_length.sh
```

Or with a smaller batch size if you hit OOM at long sequences:
```bash
!bash scripts/sweep_length.sh 2
```

---

## Step 6 — Verify checkpoints

```python
import torch, glob

for arch in ['transformer', 'tcn', 'mamba']:
    ckpts = glob.glob(f'checkpoints/{arch}/*.pt')
    assert len(ckpts) > 0, f'No checkpoint found for {arch}'
    ckpt = torch.load(f'checkpoints/{arch}/latest.pt', map_location='cpu')
    print(f'{arch:>12s}  step={ckpt["step"]:,}  val_bpc={ckpt["val_bpc"]:.4f}')
```

---

## Step 7 — Generate all figures

```python
import json
from src.viz import generate_all

with open('checkpoints/sweep_results.json') as f:
    sweeps = json.load(f)

# Optional: pre-compute eval results for bpc bar charts
# (skip if you want only the sweep/schematic figures)
from src.models import build_model
from src.dataset import get_dataloader
from src.data import load_split
from src.evaluate import evaluate, evaluate_long_range
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
val_data = load_split('val')
eval_results = {}

for arch in ['transformer', 'tcn', 'mamba']:
    ckpt  = torch.load(f'checkpoints/{arch}/latest.pt', map_location='cpu')
    model = build_model(arch, **ckpt['config']['model'])
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    dl    = get_dataloader('val', seq_len=ckpt['config']['seq_len'],
                           batch_size=8, num_workers=0)
    std   = evaluate(model, dl)
    lr    = evaluate_long_range(model, val_data, seq_len=ckpt['config']['seq_len'])
    eval_results[arch] = {**std, **lr}
    print(f'{arch}: bpc={std["bpc"]:.4f}  long_range_bpc={lr["long_range_bpc"]:.4f}')

saved = generate_all(
    checkpoints={
        'transformer': 'checkpoints/transformer/latest.pt',
        'tcn':         'checkpoints/tcn/latest.pt',
        'mamba':       'checkpoints/mamba/latest.pt',
    },
    sweep_results='checkpoints/sweep_results.json',
    output_dir='figures/',
    eval_results=eval_results,
    device=device,
)
for name, path in saved.items():
    print(f'{name}: {path}')
```

---

## Step 8 — Run the notebooks

Open and run cell-by-cell:

```bash
# Post A: What the Transformer Assumes
notebooks/post_A.ipynb

# Post B: What Structure Buys You (Mamba)
notebooks/post_B.ipynb
```

Both notebooks are self-contained — they re-run evaluation and regenerate figures inline.  
Cell 2 of each notebook is the only place you need to update paths if checkpoints are in a non-default location.

---

## Saving checkpoints to Kaggle persistent storage

Kaggle notebook sessions are ephemeral. To avoid re-training, save `checkpoints/` to a Kaggle dataset:

```python
# After training completes, copy to /kaggle/working/ (auto-persisted as output)
import shutil
shutil.copytree('checkpoints', '/kaggle/working/checkpoints')
```

On subsequent sessions, load from the saved dataset:
```python
import shutil
shutil.copytree('/kaggle/input/your-dataset-name/checkpoints', 'checkpoints')
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'src'` | Run `%cd transformer-vs-ssm` or add repo root to `sys.path` |
| OOM at seq_len=4096 | Lower batch size: `--batch_size 4` or `bash scripts/sweep_length.sh 1` |
| Training resumes from wrong checkpoint | Delete `checkpoints/{arch}/latest.pt` to force a fresh start |
| `seaborn` not found | `!pip install seaborn` |
| Figures directory missing | Created automatically by `generate_all()` and the viz functions |
