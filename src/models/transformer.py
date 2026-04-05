"""
Causal autoregressive Transformer for byte-level language modelling.

Architecture choices (all per the project spec):
- Pre-LayerNorm (LN before attention and FFN, not after)
- Learned positional embeddings
- Weight tying: output projection shares weights with token embedding
- Standard scaled dot-product attention with lower-triangular causal mask
  (no FlashAttention — memory scaling is deliberately unoptimised)
- Optional return of attention weights for visualisation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Lower-triangular causal mask — registered as a buffer (not a parameter)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to (B, n_heads, T, d_head)
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale          # (B, H, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))

        if return_attention:
            return out, attn
        return out, None


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Pre-LN: normalise before each sub-layer, add residual after
        attn_out, attn_weights = self.attn(self.ln1(x), return_attention=return_attention)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, attn_weights


class Transformer(nn.Module):
    """Causal byte-level Transformer.

    Args:
        vocab_size:  number of token types (256 for raw bytes)
        n_layers:    number of Transformer blocks
        d_model:     embedding / residual stream dimension
        n_heads:     number of attention heads
        d_ff:        feed-forward hidden dimension
        dropout:     dropout probability (applied in attention and FFN)
        max_seq_len: maximum sequence length (for positional embeddings and mask)
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection — weight-tied to token embedding
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x:                LongTensor of shape (batch, seq_len)
            return_attention: if True, also return list of attention weight tensors

        Returns:
            logits:          FloatTensor of shape (batch, seq_len, vocab_size)
            attention_weights: list of (B, n_heads, T, T) tensors per layer,
                               or None if return_attention is False
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        h = self.drop(self.tok_emb(x) + self.pos_emb(positions))

        all_attn = [] if return_attention else None
        for block in self.blocks:
            h, attn_weights = block(h, return_attention=return_attention)
            if return_attention:
                all_attn.append(attn_weights)

        h = self.ln_f(h)
        logits = self.head(h)  # (B, T, vocab_size)

        if return_attention:
            return logits, all_attn
        return logits
