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

MAMBA_SCAN = "parallel-associative"
print(f"[mamba.py] scan='{MAMBA_SCAN}' (Hillis-Steele, no Python loop over T)")


def _associative_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Parallel prefix scan for the linear recurrence h_t = a_t * h_{t-1} + b_t, h_0 = 0.

    Implements the Hillis-Steele inclusive prefix scan over affine maps.
    Each element is a pair (a, b) representing h → a*h + b.
    Composition (left applied first, then right):
        (a_L, b_L) ∘ (a_R, b_R)  =  (a_L * a_R,  a_R * b_L + b_R)

    Args:
        a: (B, T, E, N) — A_bar values, should be in (0, 1) for numerical stability
        b: (B, T, E, N) — Bu values (input terms)

    Returns:
        h: (B, T, E, N) — hidden states at every position
    """
    B, T, E, N = a.shape
    acc_a = a.clone()
    acc_b = b.clone()

    step = 1
    while step < T:
        # Shift right by `step`: pad with identity element (1, 0) on the left
        left_a = torch.cat([torch.ones( B, step, E, N, device=a.device, dtype=a.dtype),
                            acc_a[:, :-step]], dim=1)
        left_b = torch.cat([torch.zeros(B, step, E, N, device=a.device, dtype=a.dtype),
                            acc_b[:, :-step]], dim=1)
        # Compose: (left_a, left_b) ∘ (acc_a, acc_b)
        # = (left_a * acc_a,  acc_a * left_b + acc_b)
        new_a = left_a * acc_a
        acc_b = acc_a * left_b + acc_b   # use original acc_a before overwriting
        acc_a = new_a
        step *= 2

    return acc_b   # h_t = B_1t when h_0 = 0


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

        # ── 4. Discretise A and compute Bu — fully vectorised ────────────────
        # A is (E, N), negative real
        A = -torch.exp(self.A_log)                  # (E, N)

        # A_bar_t = exp(delta_t * A)  ∈ (0, 1)  — shape (B, T, E, N)
        A_bar = torch.exp(delta.unsqueeze(-1) * A)   # (B, T, E, N)

        # Bu_t = B_bar_t * x_t  — shape (B, T, E, N)
        Bu = (delta.unsqueeze(-1)                    # (B, T, E, 1)
              * B_ssm.unsqueeze(2)                   # (B, T, 1, N)
              * h_conv.unsqueeze(-1))                # (B, T, E, 1)

        # ── 5. Numerically stable parallel prefix scan ────────────────────────
        # Solves h_t = A_bar_t * h_{t-1} + Bu_t for all t simultaneously.
        #
        # Uses the Hillis-Steele associative scan over pairs (a, b) representing
        # the affine map h → a*h + b, with composition:
        #   (a_left, b_left) ∘ (a_right, b_right) = (a_left*a_right, a_right*b_left + b_right)
        #
        # A_bar ∈ (0,1) at every step, so products never overflow — unlike the
        # naive cumsum-of-logs approach which loses precision at long sequences.
        # O(T log T) work across log(T) sequential tensor ops (no Python loop over T).

        h = _associative_scan(A_bar, Bu)            # (B, T, E, N)

        # Output: y_t = sum_N(C_t * h_t)
        y = (h * C_ssm.unsqueeze(2)).sum(-1)         # (B, T, E)

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
