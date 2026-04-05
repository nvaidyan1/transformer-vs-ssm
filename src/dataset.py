"""
PyTorch Dataset and DataLoader for byte-sequence next-byte prediction.

Sequences are drawn by striding through the data with no gaps — stride equals
seq_len so every byte is covered exactly once per epoch with no overlap.
This is deterministic and reproducible with no random sampling.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.data import load_split


class ByteSequenceDataset(Dataset):
    """Fixed-stride byte-sequence dataset for next-byte prediction.

    Args:
        data:    uint8 numpy array of raw bytes.
        seq_len: length of each input/target sequence.

    Each item returns (x, y) where:
        x: LongTensor of shape (seq_len,) — input bytes
        y: LongTensor of shape (seq_len,) — targets = x shifted right by one
    """

    def __init__(self, data: np.ndarray, seq_len: int) -> None:
        self.data = data
        self.seq_len = seq_len
        # Number of full non-overlapping windows where we can still read
        # seq_len+1 bytes (needed for the target's last position).
        self.n_seqs = (len(data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_seqs

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


def get_dataloader(
    split: str,
    seq_len: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader for the given split.

    Args:
        split:       'train' or 'val'
        seq_len:     sequence length for each sample
        batch_size:  number of sequences per batch
        num_workers: DataLoader worker processes
        shuffle:     whether to shuffle (default False — stride order is canonical)

    Returns:
        DataLoader yielding (x, y) pairs of shape (batch_size, seq_len)
    """
    data = load_split(split)
    dataset = ByteSequenceDataset(data, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
