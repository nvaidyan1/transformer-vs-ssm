"""
Causal Dilated Convolutional Network (TCN) for byte-level language modelling.

Architecture choices (per project spec):
- Dilation doubles each layer: dilation = 2^i at layer i
- Causal padding: pad (kernel_size-1)*dilation zeros on the left only
- Residual connections with 1x1 convolution for dimension matching
- Weight norm on all convolutional layers
- Receptive field = 1 + (kernel_size - 1) * sum(2^i for i in range(n_layers))
"""

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


class TCN(nn.Module):
    """Causal dilated TCN for byte-level language modelling.

    Args:
        vocab_size:  number of token types (256 for raw bytes)
        n_layers:    number of dilated conv blocks
        d_model:     channel dimension throughout the network
        kernel_size: convolution kernel size (same for all layers)
        dropout:     dropout probability

    Receptive field (in bytes):
        1 + (kernel_size - 1) * sum(2^i for i in range(n_layers))
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TCNBlock(d_model, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        self.receptive_field = 1 + (kernel_size - 1) * sum(2 ** i for i in range(n_layers))

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
