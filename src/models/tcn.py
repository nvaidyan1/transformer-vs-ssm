"""
Causal Dilated Convolutional Network (TCN) for byte-level language modelling.

Architecture choices (per project spec):
- Dilation doubles each layer, cycling every `dilation_cycle` layers:
  dilation = 2^(i % dilation_cycle) at layer i
- Causal padding: pad (kernel_size-1)*dilation zeros on the left only
- Residual connections with 1x1 convolution for dimension matching
- Weight norm on all convolutional layers

Why the dilation cycles (WaveNet-style stacks)
----------------------------------------------
An uncapped `dilation = 2^i` schedule grows the left padding exponentially.
With n_layers=18, kernel_size=7 and seq_len=1024 the final layer pads
(7-1) * 2^17 = 786,432 zeros onto a 1,024-byte sequence, producing a single
(16, 256, 787456) fp32 activation of 12.02 GiB — an immediate OOM on a 16 GB
GPU, and 24.3 GiB across all 18 layers.

It is also pointless. Eight layers already give a receptive field of 1,531
bytes, which covers the full 1,024-byte context. Everything past layer 8 is
convolving over padded zeros: the nominal receptive field reaches 1,572,859
bytes, 1,536x the sequence length.

Cycling the dilation back to 1 every `dilation_cycle` layers keeps the full
context covered (RF = 3,079 for 18 layers at cycle 8), caps the padding at
(kernel_size-1) * 2^(cycle-1) = 768, and cuts total padding memory 74x to
336 MiB. Parameter count is unaffected — dilation appears in no weight shape —
so the ~10M-parameter three-way comparison is unchanged.

Choose dilation_cycle as the smallest c with 1 + (kernel_size-1)*(2^c - 1)
>= seq_len. For kernel_size=7, seq_len=1024 that is c=8.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """Conv1d with left-only (causal) padding and weight normalisation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """One dilated causal conv block with a residual connection.

    The residual path uses a 1x1 conv for dimension matching (always present
    for architectural consistency, acts as identity when channels are equal).
    """

    def __init__(self, d_model: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv = CausalConv1d(d_model, d_model, kernel_size, dilation)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # 1x1 residual projection (weight-normed per spec)
        self.res_conv = weight_norm(nn.Conv1d(d_model, d_model, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        # LayerNorm expects (B, T, C) — transpose in and out
        h = self.norm(x.transpose(1, 2)).transpose(1, 2)
        h = self.conv(h)
        h = self.act(h)
        h = self.drop(h)
        return h + self.res_conv(x)


def required_dilation_cycle(seq_len: int, kernel_size: int) -> int:
    """Smallest cycle length whose receptive field covers `seq_len`.

    Returns the smallest c such that 1 + (kernel_size-1)*(2^c - 1) >= seq_len.
    """
    c = 1
    while 1 + (kernel_size - 1) * (2 ** c - 1) < seq_len:
        c += 1
    return c


class TCN(nn.Module):
    """Causal dilated TCN for byte-level language modelling.

    Args:
        vocab_size:     number of token types (256 for raw bytes)
        n_layers:       number of dilated conv blocks
        d_model:        channel dimension throughout the network
        kernel_size:    convolution kernel size (same for all layers)
        dropout:        dropout probability
        dilation_cycle: dilation resets to 1 every this many layers. Layer i
                        uses dilation 2^(i % dilation_cycle). Set to n_layers
                        to recover the old uncapped 2^i schedule (will OOM for
                        n_layers > ~12 at d_model=256, batch=16, seq_len=1024).

    Receptive field (in bytes):
        1 + (kernel_size - 1) * sum(2^(i % dilation_cycle) for i in range(n_layers))
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        kernel_size: int,
        dropout: float,
        dilation_cycle: int = 8,
    ):
        super().__init__()
        if dilation_cycle < 1:
            raise ValueError(f"dilation_cycle must be >= 1, got {dilation_cycle}")

        self.dilation_cycle = dilation_cycle
        self.dilations = [2 ** (i % dilation_cycle) for i in range(n_layers)]

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size, dilation=d, dropout=dropout)
            for d in self.dilations
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.receptive_field = 1 + (kernel_size - 1) * sum(self.dilations)
        self.max_padding = (kernel_size - 1) * max(self.dilations)

        print(f"[tcn.py] n_layers={n_layers} kernel_size={kernel_size} "
              f"dilation_cycle={dilation_cycle} "
              f"max_dilation={max(self.dilations)} "
              f"max_left_pad={self.max_padding} "
              f"receptive_field={self.receptive_field:,}")

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor of shape (batch, seq_len)

        Returns:
            logits: FloatTensor of shape (batch, seq_len, vocab_size)
        """
        h = self.tok_emb(x).transpose(1, 2)  # (B, d_model, T)

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h.transpose(1, 2))     # (B, T, d_model)
        return self.head(h)                   # (B, T, vocab_size)
