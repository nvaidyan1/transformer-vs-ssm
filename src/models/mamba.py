"""
Pure-PyTorch Selective State Space Model (Mamba) for byte-level language modelling.

Architecture choices (per project spec):
- No Triton kernels, no mamba-ssm package — plain PyTorch throughout
- Input-dependent delta_t, B_t, C_t projected from each token
- A initialised as diagonal negative reals, parameterised in log space (stable by construction)
- Sequential scan in the forward pass for clarity
  (production Mamba uses a parallel associative scan — O(T log T) vs O(T) here,
   but the two are mathematically identical)
- Optional return of delta_t across the sequence for visualisation

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaSSM(nn.Module):
    """
    The selective SSM mixer at the core of each Mamba block.

    Shapes throughout (B=batch, T=seq_len, E=d_inner, N=d_state):
        in_proj  : (B, T, d_model) → (B, T, 2*E)   — x and gate z
        conv1d   : (B, E, T)       → (B, E, T)      — causal depthwise conv
        x_proj   : (B, T, E)       → (B, T, E+2*N)  — delta, B, C
        SSM scan : runs T steps, state h ∈ ℝ^(B, E, N)
        out_proj : (B, T, E)       → (B, T, d_model)
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.d_conv = d_conv

        # Project input to x and gate in one matmul
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv (groups=d_inner makes it channel-wise)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,   # we'll trim the right side to keep causality
            groups=self.d_inner,
            bias=True,
        )

        # Project x → (delta, B, C) — delta has same width as d_inner for clarity
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * d_state, bias=False)

        # A: (d_inner, d_state) — stored as log so exp(A_log) > 0,
        # and we negate to get stable negative-real diagonal eigenvalues
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A_init = A_init.expand(self.d_inner, -1).log()   # log of 1..d_state
        self.A_log = nn.Parameter(A_init)

        # D: skip-connection weight (one per inner channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, return_delta: bool = False):
        """
        Args:
            x:            (B, T, d_model)
            return_delta: if True, also return delta_t for visualisation

        Returns:
            out:   (B, T, d_model)
            delta: (B, T, d_inner) or None
        """
        B, T, _ = x.shape

        # ── 1. Input projection ──────────────────────────────────────────────
        xz = self.in_proj(x)                        # (B, T, 2*E)
        x_in, z = xz.chunk(2, dim=-1)               # each (B, T, E)

        # ── 2. Causal local conv ─────────────────────────────────────────────
        # Conv1d expects (B, C, T); trim right padding to keep causality
        h_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T]   # (B, E, T)
        h_conv = F.silu(h_conv).transpose(1, 2)                 # (B, T, E)

        # ── 3. SSM projections ───────────────────────────────────────────────
        dBC = self.x_proj(h_conv)                   # (B, T, E + 2*N)
        delta = dBC[:, :, : self.d_inner]           # (B, T, E)
        B_ssm = dBC[:, :, self.d_inner : self.d_inner + self.d_state]  # (B, T, N)
        C_ssm = dBC[:, :, self.d_inner + self.d_state :]               # (B, T, N)

        delta = F.softplus(delta)                   # enforce positivity  (B, T, E)

        # ── 4. Discretise A ──────────────────────────────────────────────────
        # A is (E, N), negative real; keep it fixed-shape for the scan
        A = -torch.exp(self.A_log)                  # (E, N)

        # ── 5. Sequential selective scan ─────────────────────────────────────
        # State h: (B, E, N)
        state = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(T):
            dt = delta[:, t, :]                     # (B, E)
            b_t = B_ssm[:, t, :]                    # (B, N)
            c_t = C_ssm[:, t, :]                    # (B, N)
            x_t = h_conv[:, t, :]                   # (B, E)

            # Discretise: zero-order hold
            # A_bar: (B, E, N)  —  exp(dt * A)
            A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))   # (B, E, N)
            # B_bar: (B, E, N)  —  dt * b_t broadcast
            B_bar = dt.unsqueeze(-1) * b_t.unsqueeze(1)            # (B, E, N)

            # State update: h = A_bar * h + B_bar * x_t
            state = A_bar * state + B_bar * x_t.unsqueeze(-1)      # (B, E, N)

            # Output: y_t = sum_N(C * h)
            y_t = (state * c_t.unsqueeze(1)).sum(-1)               # (B, E)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                  # (B, T, E)

        # ── 6. Skip connection + gate ─────────────────────────────────────────
        y = y + h_conv * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)

        # ── 7. Output projection ─────────────────────────────────────────────
        out = self.out_proj(y)                      # (B, T, d_model)

        return out, delta if return_delta else None


class MambaBlock(nn.Module):
    """Pre-LN residual block wrapping one MambaSSM mixer."""

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = MambaSSM(d_model, d_state, d_conv, expand)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_delta: bool = False):
        h, delta = self.ssm(self.norm(x), return_delta=return_delta)
        return x + self.drop(h), delta


class Mamba(nn.Module):
    """Selective SSM (Mamba) for byte-level language modelling.

    Args:
        vocab_size: number of token types (256 for raw bytes)
        n_layers:   number of Mamba blocks
        d_model:    model/embedding dimension
        d_state:    SSM state dimension (N in the paper)
        d_conv:     width of the local causal depthwise conv
        expand:     inner expansion factor; d_inner = d_model * expand
        dropout:    dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_delta: bool = False):
        """
        Args:
            x:            LongTensor of shape (batch, seq_len)
            return_delta: if True, also return delta_t from every block

        Returns:
            logits:     FloatTensor of shape (batch, seq_len, vocab_size)
            all_deltas: list of (batch, seq_len, d_inner) tensors per layer,
                        or None if return_delta is False
        """
        h = self.tok_emb(x)                        # (B, T, d_model)

        all_deltas = [] if return_delta else None
        for block in self.blocks:
            h, delta = block(h, return_delta=return_delta)
            if return_delta:
                all_deltas.append(delta)

        h = self.ln_f(h)
        logits = self.head(h)

        if return_delta:
            return logits, all_deltas
        return logits
