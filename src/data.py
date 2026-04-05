"""
enwik8 download, byte encoding, and train/val split.

Splits: first 90,000,000 bytes → train, remainder → val.
Saved as raw uint8 .bin files under data/.
"""

import os
import zipfile
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
URL = "http://mattmahoney.net/dc/enwik8.zip"
TRAIN_BYTES = 90_000_000


def prepare_data() -> None:
    """Download enwik8 if needed, split into train/val, save as .bin files."""
    DATA_DIR.mkdir(exist_ok=True)

    train_path = DATA_DIR / "train.bin"
    val_path = DATA_DIR / "val.bin"

    if train_path.exists() and val_path.exists():
        return

    zip_path = DATA_DIR / "enwik8.zip"
    raw_path = DATA_DIR / "enwik8"

    if not raw_path.exists():
        if not zip_path.exists():
            print(f"Downloading enwik8 from {URL} ...")
            urllib.request.urlretrieve(URL, zip_path)
        print("Unzipping enwik8 ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("enwik8", DATA_DIR)

    print("Reading raw bytes ...")
    data = np.frombuffer(raw_path.read_bytes(), dtype=np.uint8)

    train = data[:TRAIN_BYTES]
    val = data[TRAIN_BYTES:]

    train.tofile(train_path)
    val.tofile(val_path)
    print(f"Saved train ({len(train):,} bytes) and val ({len(val):,} bytes) to {DATA_DIR}")


def load_split(split: str) -> np.ndarray:
    """Return the requested split as a uint8 numpy array.

    Args:
        split: 'train' or 'val'

    Returns:
        np.ndarray of dtype uint8
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    path = DATA_DIR / f"{split}.bin"
    if not path.exists():
        prepare_data()

    return np.fromfile(path, dtype=np.uint8)
