# Kaggle Training Checklist

Follow this top to bottom. Each section is a gate — do not proceed until it passes.

---

## Before you open a session

- [ ] Kaggle notebook settings: **Accelerator → GPU T4 x1**, internet ON, persistence ON
- [ ] Confirm quota: **20 GB** on `/kaggle/working`. You need at least 4 GB free before starting.
  - enwik8 data: ~200 MB
  - Checkpoints (keep_last=2 per model × 3 models × ~150 MB): ~900 MB
  - Safety margin for AMP scaler state, logs, output zip: ~500 MB
  - Total in use: **~1.6 GB** — well within quota with the fixes in place

---

## Session plan by model

Mamba's sequential scan is ~25× slower than the transformer on T4.
Plan your sessions accordingly — do not start a model you cannot finish or resume.

| Model | ms/step | 50k steps | Val eval overhead | Est. total |
|-------|---------|-----------|-------------------|------------|
| Transformer | ~61 ms | ~51 min | ~25 min | **~1.5 h** |
| TCN | ~50 ms | ~42 min | ~25 min | **~1.5 h** |
| Mamba | ~1600 ms | ~22 h | ~25 min | **~23 h** |

**Mamba requires multiple sessions.** The train script resumes automatically from the latest checkpoint — just re-run the same command and it picks up where it left off.

Kaggle GPU session limit: **9 h** (interactive) / **12 h** (commit run).
Suggested plan: transformer + TCN in one session, Mamba across 3 sessions.

---

## Step 1 — One-time setup (first session only)

```python
# Cell 1: clone and enter repo
import os, subprocess, sys

if not os.path.exists('transformer-vs-ssm'):
    subprocess.run(['git', 'clone',
                    'https://github.com/nvaidyan1/transformer-vs-ssm.git'], check=True)

os.chdir('transformer-vs-ssm')
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

print(os.getcwd())
```

```python
# Cell 2: install dependencies
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'torch', 'numpy', 'pyyaml', 'matplotlib', 'seaborn', 'tqdm'], check=True)
print('deps OK')
```

```python
# Cell 3: download and split enwik8 (~200 MB, runs once)
from src.data import prepare_data
prepare_data()
```

- [ ] "Train: 90,000,000 bytes" printed
- [ ] "Val: 10,000,000 bytes" printed

---

## Step 2 — Pull latest and run test suite (every session)

Always do this at the start of each session to pick up any fixes pushed since the last run.

```python
# Cell 4: pull latest
subprocess.run(['git', 'pull'], check=True)
```

```python
# Cell 5: run checkpoint safety tests — must be 9/9 before training
result = subprocess.run(
    [sys.executable, 'tests/test_checkpoint_fixes.py'],
    capture_output=True, text=True
)
print(result.stdout[-2000:])   # last 2000 chars — contains the summary line
print(result.stderr[-500:] if result.returncode != 0 else '')
assert result.returncode == 0, 'Tests failed — do not train until fixed'
```

- [ ] `Ran 9 tests in ... OK` printed
- [ ] Return code 0

---

## Step 3 — Verify disk headroom

```python
# Cell 6
import shutil
free_gb = shutil.disk_usage('/kaggle/working').free / 1024**3
print(f'Free: {free_gb:.1f} GB')
assert free_gb > 4.0, f'Not enough disk: {free_gb:.1f} GB — clear space first'
```

- [ ] At least 4 GB free reported

---

## Step 4 — Train transformer (~1.5 h)

```python
# Cell 7
subprocess.run([sys.executable, 'src/train.py',
                '--config', 'configs/transformer.yaml'], check=True)
```

**Watch for:**
- `[mamba.py] scan='sequential'...` on import — confirms you have the latest mamba.py
- `disk free : X.X GB` in the config block — confirms Fix 1 is active
- Log lines every 500 steps: `step   500 | lr ... | train_bpc ... | val_bpc ...`
- val_bpc should drop from ~8 toward ~1.5–1.8 over 50k steps
- `checkpoint → checkpoints/transformer/ckpt_XXXXXXX.pt` every 1000 steps
- At most 2 `.pt` files in `checkpoints/transformer/` at any time (Fix 3)

**If it crashes mid-run:** re-run Cell 7. The script detects `latest.pt` and resumes automatically. You will see `Resumed from ... (step XXXXX, val_bpc X.XXXX)`.

- [ ] `Training complete. Final checkpoint →` printed
- [ ] `checkpoints/transformer/latest.pt` exists

---

## Step 5 — Verify transformer checkpoint

```python
# Cell 8
import torch, glob

ckpt = torch.load('checkpoints/transformer/latest.pt',
                  map_location='cpu', weights_only=False)
print(f'step:    {ckpt["step"]:,}')
print(f'val_bpc: {ckpt["val_bpc"]:.4f}')
assert ckpt['step'] == 50000, f'Expected 50000 steps, got {ckpt["step"]}'
assert ckpt['val_bpc'] < 2.5, f'val_bpc suspiciously high: {ckpt["val_bpc"]:.4f}'
print('Transformer checkpoint OK')
```

- [ ] step = 50,000
- [ ] val_bpc < 2.5

---

## Step 6 — Train TCN (~1.5 h)

```python
# Cell 9
subprocess.run([sys.executable, 'src/train.py',
                '--config', 'configs/tcn.yaml'], check=True)
```

**Watch for:**
- `FutureWarning` about `weight_norm` — harmless, expected
- Same log pattern as transformer
- TCN memory is high (~3 GB on forward pass) but T4 has 14.6 GB — fine

- [ ] `Training complete. Final checkpoint →` printed
- [ ] `checkpoints/tcn/latest.pt` exists

---

## Step 7 — Save checkpoints before session ends

**Do this before your session times out, even if Mamba is not done.**

```python
# Cell 10: zip transformer + TCN checkpoints to /kaggle/working (persisted as output)
import zipfile, shutil
from pathlib import Path

zip_path = Path('/kaggle/working/checkpoints_partial.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for arch in ['transformer', 'tcn', 'mamba']:
        for f in sorted(Path(f'checkpoints/{arch}').rglob('*.pt')):
            zf.write(f)

size_mb = zip_path.stat().st_size / 1024**2
n_pts = sum(1 for _ in Path('checkpoints').rglob('*.pt'))
print(f'Zipped {n_pts} checkpoints → {zip_path} ({size_mb:.1f} MB)')
print('Download via: Kaggle notebook Output tab → Download')
```

- [ ] Zip file created and size is reasonable
- [ ] Downloaded locally or noted in Output tab

---

## Step 8 — Train Mamba (multiple sessions, ~23 h total)

Mamba requires 3 sessions of ~7–8 h each on T4. The resume logic handles this automatically.

**Each session:**

```python
# Cell 11 (repeat each session after Steps 1–3)
subprocess.run([sys.executable, 'src/train.py',
                '--config', 'configs/mamba.yaml'], check=True)
```

**Session 1:** runs from step 0 → ~16,000 (at ~1600ms/step over ~7 h)
**Session 2:** resumes from ~16,000 → ~32,000
**Session 3:** resumes from ~32,000 → 50,000

At the start of each session you will see:
```
Resumed from checkpoints/mamba/latest.pt (step 16000, val_bpc X.XXXX)
```

**If you lose the Kaggle session before zipping:**
The output from a completed/saved session is available under the notebook's Output tab.
Download `checkpoints_partial.zip` and re-upload as a Kaggle dataset to restore.

**Restoring from a Kaggle dataset:**
```python
import shutil
shutil.copytree('/kaggle/input/your-dataset-name/checkpoints', 'checkpoints')
```

- [ ] Session 1 complete, step ≥ 15,000
- [ ] Session 2 complete, step ≥ 30,000
- [ ] Session 3 complete, step = 50,000

---

## Step 9 — Final verification (after all three models)

```python
# Cell 12
import torch
from pathlib import Path

print(f'{"arch":>12s}  {"step":>8s}  {"val_bpc":>8s}  {"ckpt"}')
print('-' * 60)
for arch in ['transformer', 'tcn', 'mamba']:
    p = Path(f'checkpoints/{arch}/latest.pt')
    assert p.exists(), f'Missing: {p}'
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    print(f'{arch:>12s}  {ckpt["step"]:>8,}  {ckpt["val_bpc"]:>8.4f}  {p.resolve().name}')
    assert ckpt['step'] == 50000
    assert ckpt['val_bpc'] < 2.5
print('\nAll three models trained and verified.')
```

- [ ] All three at step 50,000
- [ ] All three val_bpc < 2.5

---

## Step 10 — Run full sweep (generates sweep_results.json for notebooks)

```python
# Cell 13
subprocess.run(['bash', 'scripts/sweep_length.sh'], check=True)
```

- [ ] `checkpoints/sweep_results.json` created
- [ ] Results printed for all three models at all six seq_lengths

---

## Step 11 — Final save

```python
# Cell 14: final zip of all checkpoints + sweep results
import zipfile
from pathlib import Path

zip_path = Path('/kaggle/working/checkpoints_final.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(Path('checkpoints').rglob('*.pt')):
        zf.write(f)
    sweep = Path('checkpoints/sweep_results.json')
    if sweep.exists():
        zf.write(sweep)

size_mb = zip_path.stat().st_size / 1024**2
print(f'Final zip: {zip_path} ({size_mb:.1f} MB)')
```

- [ ] `checkpoints_final.zip` created and downloaded

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `UnboundLocalError: num_workers` | Old code on disk | `git pull` then re-run |
| `Disk headroom too low` | < 2 GB free | Delete old output zips from `/kaggle/working` |
| `RuntimeError: unexpected pos` | Torn write (old code) | `git pull` — atomic writes prevent this now |
| `OSError: [Errno 28] No space left` | Quota exhausted | Zip and clear checkpoints, restart session |
| `Resumed from ... step 0` | `latest.pt` symlink broken | `ls checkpoints/{arch}/` to find highest-numbered `.pt`, then `ln -sf ckpt_XXXXXXX.pt checkpoints/{arch}/latest.pt` |
| Mamba `val_bpc` not dropping | Still in warmup | Normal — warmup is 2000 steps. Wait until step 2000 before judging. |
| TCN `FutureWarning: weight_norm` | PyTorch deprecation | Harmless — ignore |
| `assert result.returncode == 0` | Test suite failed | Read the test output above the assertion, fix the issue in train.py |
