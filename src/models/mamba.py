"""
Pure-PyTorch Selective State Space Model (Mamba) for byte-level language modelling.

Architecture choices (per project spec):
- No Triton kernels, no mamba-ssm package — plain PyTorch throughout
- Input-dependent delta_t, B_t, C_t projected from each token
- A initialised as diagonal negative reals, parameterised in log space (stable by construction)
- Sequential scan in the forward pass (O(n) memory, O(n) FLOPs)
  Production Mamba uses Triton-fused parallel associative scan for wall-clock speed;
  without custom kernels, the parallel scan in plain PyTorch uses more memory and
  is slower due to tensor allocation overhead — so we keep the sequential scan here.
  The honest efficiency claim is O(n) memory (vs transformer O(n²)) and O(1) memory
  at inference (recurrent mode, one token at a time).
- Optional return of delta_t across the sequence for visualisation

Reference: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Delta (the selection timestep) must be initialised small, per Gu & Dao's reference
# implementation, or the state decays to nothing within a few steps regardless of
# what the model learns. This is not a tuning nicety -- with bias-free projections
# and default nn.Linear init, softplus(pre-activation) lands at softplus(0) = ln(2)
# = 0.693, and A initialised at -1..-16 gives a state half-life of ~1 STEP. A model
# built this way cannot retain information across a 64-token sequence no matter how
# it trains; the Selective Copying gate confirmed this (mean acc 0.022, at chance).
#
# Fix: a dedicated dt_proj whose bias is initialised so softplus(bias) is log-uniform
# in [DT_MIN, DT_MAX] BEFORE any input-dependent contribution is added (see
# MambaSSM.__init__). These are Gu & Dao's defaults.
DT_MIN = 0.001
DT_MAX = 0.1

# Default scan implementation. "sequential" is a Python loop over timesteps;
# "parallel" uses the Hillis-Steele prefix scan in _associative_scan below.
#
# Which is faster depends on shape, and the crossover is sharp (measured by
# scripts/bench_mamba_scan.py at d_inner=512, d_state=16):
#
#     device  batch  seq_len   sequential   parallel   result
#     MPS        16       64       5.7 ms     1.6 ms   3.5x faster
#     MPS        16      128      14.3 ms     2.1 ms   6.9x faster
#     MPS        16      256      30.0 ms   120.7 ms   4.0x slower
#     MPS        16     1024     102.6 ms   729.0 ms   7.1x slower
#
# The parallel scan materialises O(B*T*d_inner*d_state) tensors log2(T) times, so it
# wins at short sequences (few kernel launches) and loses badly at long ones (memory
# traffic). Synthetic-task runs are short and should set scan="parallel"; the enwik8
# configuration is long and keeps the sequential default.
#
# The two are numerically equivalent to float32 rounding (~1e-7); tests/test_mamba_scan.py
# asserts identical model outputs.
MAMBA_SCAN = "sequential"

# Recompute each block's activations during backward instead of storing them.
#
# The sequential scan keeps ~4 (B, d_inner, d_state) tensors alive per timestep
# for the backward pass: A_bar, the exp() output, B_bar, and the new state. At
# B=16, d_inner=512, d_state=16 that is 2.0 MiB per timestep, so 1024 timesteps
# cost 2.0 GiB per layer and 14 layers need 28.0 GiB — more than a 16 GB T4.
# (Observed: OOM at 14.42 GiB allocated, roughly halfway through the stack.)
#
# Checkpointing keeps only one layer's scan live at a time, plus the 14 stored
# layer inputs (0.22 GiB), for a peak near 2.2 GiB. The trade is one extra
# forward pass through the scan during backward, roughly +30% step time.
#
# This changes no mathematics, no parameter count, and no result — only the
# memory/time trade-off. Set to False to recover the original behaviour.
MAMBA_GRAD_CHECKPOINT = True
print(f"[mamba.py] scan='{MAMBA_SCAN}' (O(n) memory, honest wall-clock — see docstring)")


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
        x_proj   : (B, T, E)       → (B, T, 2*N)    — B, C
        dt_proj  : (B, T, E)       → (B, T, E)      — delta, pre-softplus (see DT_MIN/DT_MAX)
        SSM scan : runs T steps, state h ∈ ℝ^(B, E, N)
        out_proj : (B, T, E)       → (B, T, d_model)
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 scan: str = None):
        super().__init__()
        self.scan = scan or MAMBA_SCAN
        if self.scan not in ("sequential", "parallel"):
            raise ValueError(f"scan must be 'sequential' or 'parallel', got {self.scan!r}")
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

        # Project x → (B, C). No bias: these have no equivalent of the delta collapse
        # below, since a zero-mean B/C only zeros out that step's contribution rather
        # than freezing every step's decay rate.
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state, bias=False)

        # Project x → delta (pre-softplus), with the dedicated init that keeps Delta
        # small: weight small so the input-dependent term starts near zero, bias set
        # so softplus(bias) alone is log-uniform in [DT_MIN, DT_MAX] per channel.
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        dt_init_std = self.d_inner ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(DT_MAX) - math.log(DT_MIN))
            + math.log(DT_MIN)
        )
        # inverse softplus: softplus(inv_softplus(dt)) == dt
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)
        # Mamba._init_weights() re-initialises every nn.Linear after construction
        # (a uniform normal(0, 0.02) + zero bias, applied model-wide). Left alone,
        # that silently overwrites this init and reintroduces the exact bug this
        # module exists to fix. Mark both tensors so it skips them — same
        # convention as the reference implementation's `_no_reinit`.
        self.dt_proj.weight._no_reinit = True
        self.dt_proj.bias._no_reinit = True

        # A: (d_inner, d_state) — stored as log so exp(A_log) > 0,
        # and we negate to get stable negative-real diagonal eigenvalues
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A_init = A_init.expand(self.d_inner, -1).log()   # log of 1..d_state
        self.A_log = nn.Parameter(A_init)

        # D: skip-connection weight (one per inner channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor, return_delta: bool = False,
                input_independent: bool = False):
        """
        Args:
            x:            (B, T, d_model)
            return_delta: if True, also return delta_t for visualisation
            input_independent: ABLATION. Replace delta/B/C with their sequence means,
                removing position-wise selectivity while keeping every parameter and
                shape identical. Used as the control for the Selective Copying gate
                (see src/tasks/selective_copy.py): a working selective implementation
                must beat this ablation by a wide margin.

                Note this is a *stronger* baseline than a time-invariant SSM such as
                S4 — the frozen values still depend on the sequence as a whole, just
                not on position. It isolates exactly the position-wise variation the
                scan consumes. It is not an S4 reimplementation.

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
        BC = self.x_proj(h_conv)                     # (B, T, 2*N)
        B_ssm = BC[:, :, : self.d_state]             # (B, T, N)
        C_ssm = BC[:, :, self.d_state :]              # (B, T, N)

        delta = F.softplus(self.dt_proj(h_conv))     # (B, T, E); see DT_MIN/DT_MAX above

        if input_independent:
            # Ablation: collapse the position axis so every timestep sees the same
            # selection parameters. Shapes are unchanged, so the scan below is
            # untouched and the comparison isolates selectivity alone.
            delta = delta.mean(dim=1, keepdim=True).expand_as(delta)
            B_ssm = B_ssm.mean(dim=1, keepdim=True).expand_as(B_ssm)
            C_ssm = C_ssm.mean(dim=1, keepdim=True).expand_as(C_ssm)

        # ── 4. Discretise A ──────────────────────────────────────────────────
        # A is (E, N), negative real; keep it fixed-shape for the scan
        A = -torch.exp(self.A_log)                  # (E, N)

        # ── 5. Sequential selective scan ─────────────────────────────────────
        # State h: (B, E, N)
        #
        # Wall-clock note: this Python loop is slower than the transformer at
        # all measured sequence lengths on CUDA. That is expected and honest:
        # - Memory is O(n) — Mamba's real efficiency advantage at training time
        # - Compute is also O(n) in FLOPs, but the constant factor is high in
        #   pure PyTorch (the real mamba-ssm package uses Triton kernels)
        # - At inference (one token at a time), Mamba runs as a true RNN with
        #   O(1) memory per step regardless of context length — that is the
        #   hardware-verifiable efficiency claim in Post B
        if self.scan == "parallel":
            # Same recurrence, evaluated with a prefix scan over all timesteps at once.
            #   a_t = exp(delta_t * A)            (B, T, E, N)
            #   b_t = (delta_t * B_t) * x_t       (B, T, E, N)
            a = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
            b = (delta.unsqueeze(-1) * B_ssm.unsqueeze(2)) * h_conv.unsqueeze(-1)
            h = _associative_scan(a, b)                            # (B, T, E, N)
            y = (h * C_ssm.unsqueeze(2)).sum(-1)                   # (B, T, E)
        else:
            state = torch.zeros(B, self.d_inner, self.d_state,
                                device=x.device, dtype=x.dtype)
            ys = []

            for t in range(T):
                dt = delta[:, t, :]                     # (B, E)
                b_t = B_ssm[:, t, :]                    # (B, N)
                c_t = C_ssm[:, t, :]                    # (B, N)
                x_t = h_conv[:, t, :]                   # (B, E)

                # Discretise: zero-order hold
                A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))   # (B, E, N)
                B_bar = dt.unsqueeze(-1) * b_t.unsqueeze(1)            # (B, E, N)

                state = A_bar * state + B_bar * x_t.unsqueeze(-1)      # (B, E, N)
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

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int,
                 dropout: float, scan: str = None):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = MambaSSM(d_model, d_state, d_conv, expand, scan=scan)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_delta: bool = False,
                input_independent: bool = False):
        h, delta = self.ssm(self.norm(x), return_delta=return_delta,
                            input_independent=input_independent)
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
        scan:       "sequential" (default) or "parallel" — see MAMBA_SCAN above.
                    Numerically equivalent; pick by sequence length.
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
        scan: str = None,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout, scan=scan)
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
                if not getattr(m.weight, "_no_reinit", False):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, return_delta: bool = False,
                input_independent: bool = False):
        """
        Args:
            x:            LongTensor of shape (batch, seq_len)
            return_delta: if True, also return delta_t from every block
            input_independent: ablate position-wise selectivity (see MambaSSM.forward)

        Returns:
            logits:     FloatTensor of shape (batch, seq_len, vocab_size)
            all_deltas: list of (batch, seq_len, d_inner) tensors per layer,
                        or None if return_delta is False
        """
        h = self.tok_emb(x)                        # (B, T, d_model)

        all_deltas = [] if return_delta else None

        # Checkpoint only while training and only when deltas are not needed:
        # inference has no backward graph to trade against, and the delta
        # tensors are an extra output that would be recomputed for nothing.
        use_ckpt = (
            MAMBA_GRAD_CHECKPOINT
            and self.training
            and torch.is_grad_enabled()
            and not return_delta
        )

        for block in self.blocks:
            if use_ckpt:
                # use_reentrant=False preserves RNG state across the recompute,
                # so dropout draws the same mask in both passes, and composes
                # correctly with torch.amp.autocast.
                h, delta = checkpoint(block, h, False, input_independent,
                                      use_reentrant=False)
            else:
                h, delta = block(h, return_delta=return_delta,
                                 input_independent=input_independent)
            if return_delta:
                all_deltas.append(delta)

        h = self.ln_f(h)
        logits = self.head(h)

        if return_delta:
            return logits, all_deltas
        return logits
