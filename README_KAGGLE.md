# Kaggle Training Guide

All training is designed to run as **Kaggle commit runs** (Save Version → Run All), not interactively.  
This means you close your browser, Kaggle runs the notebook in the background, and you come back to download the output.

---

## Notebooks

| Notebook | Models | Estimated runtime | Sessions needed |
|----------|--------|-------------------|----------------|
| `notebooks/kaggle_train.ipynb` | Transformer + TCN | ~3–4h | 1 |
| `notebooks/kaggle_train_mamba.ipynb` | Mamba | ~23h | 3 |

---

## Session time estimates

| Model | ms/step | 50k steps | Val eval (×10) | Total |
|-------|---------|-----------|----------------|-------|
| Transformer | ~62 ms | ~52 min | ~60 min | **~1.7h** |
| TCN | ~50 ms | ~42 min | ~60 min | **~1.7h** |
| Mamba | ~1600 ms | ~22h | ~60 min | **~23h** |

Val eval runs the full 10M-byte val set every 5000 steps (10 times per model).  
Transformer + TCN together fit comfortably in one 12h commit run.

---

## Training transformer + TCN

### Before committing

- [ ] Kaggle notebook: **Accelerator → GPU T4 x1**, Internet ON, Persistence ON
- [ ] Confirm free quota ≥ 4 GB (`shutil.disk_usage('/kaggle/working').free / 1024**3`)
- [ ] Open `notebooks/kaggle_train.ipynb` in Kaggle

### How to commit

1. Click **Save Version** (top right)
2. Select **Save and Run All (Commit)**
3. Click **Save**

Kaggle queues the run. You can close the browser — the run continues in the background.

### Checking progress

- **Output tab** (notebook sidebar) — shows stdout as cells complete
- Transformer log lines appear at step 5000 (~11 min after training starts):
  ```
  step   5000 | lr 2.96e-04 | train_bpc X.XXXX | val_bpc X.XXXX | XXXs
  ```
- **Expected val_bpc trajectory:** ~5.3 at step 0 → ~2.5 at step 10k → ~1.6–1.8 at step 50k

### After the run completes

- [ ] Output tab shows `Transformer checkpoint OK` and `TCN checkpoint OK`
- [ ] `checkpoints_transformer_tcn.zip` appears in the Output tab
- [ ] Download the zip

---

## Training Mamba (3 sessions)

Mamba requires 3 commit runs of ~8h each. The resume is automatic — the script detects `latest.pt` and continues from there.

### Session 1 (step 0 → ~16,000)

1. Open `notebooks/kaggle_train_mamba.ipynb`
2. Leave `PREV_DATASET = None` in Cell 3b
3. Commit and run
4. After completion: download `mamba_checkpoint_step0016000.zip` from Output tab

### Session 2 (step ~16,000 → ~32,000)

1. Upload `mamba_checkpoint_step0016000.zip` as a new Kaggle dataset (+ New Dataset)
2. In the notebook, attach that dataset (Add Data)
3. In Cell 3b, set: `PREV_DATASET = '/kaggle/input/<your-dataset-name>'`
4. Commit and run

### Session 3 (step ~32,000 → 50,000)

Same as Session 2, using the Session 2 output zip.

### Completion check

After Session 3, the output panel should show:
```
Checkpoint: step=50,000  val_bpc=X.XXXX
```
val_bpc should be < 2.5. Download the final zip.

---

## After all three models are trained

Once you have checkpoints for all three models, run the length sweep to generate figures data:

```python
# In a new Kaggle notebook or interactive session:
import subprocess, sys
subprocess.run(['bash', 'scripts/sweep_length.sh'], check=True)
```

Then zip everything for the blog post notebooks:

```python
import zipfile
from pathlib import Path

zip_path = Path('/kaggle/working/checkpoints_final.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(Path('checkpoints').rglob('*.pt')):
        zf.write(f)
    sweep = Path('checkpoints/sweep_results.json')
    if sweep.exists():
        zf.write(sweep)

print(f'{zip_path.stat().st_size / 1024**2:.1f} MB')
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Config block never appears | `subprocess` stdout buffered | Already fixed — `-u` flag forces unbuffered output |
| `Disk headroom too low` | < 2 GB free | Delete old output zips from `/kaggle/working` |
| `RuntimeError: unexpected pos` | Torn write (old code) | `git pull` — atomic writes prevent this now |
| `OSError: [Errno 28] No space left` | Quota exhausted | Zip and clear checkpoints, restart session |
| `Resumed from ... step 0` | `latest.pt` symlink broken | Cell 3b in the Mamba notebook recreates it from the dataset |
| Mamba `val_bpc` not dropping | Still in warmup | Warmup is 2000 steps — wait until step 5000 first log line |
| TCN `FutureWarning: weight_norm` | PyTorch deprecation | Harmless — ignore |
