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

# Two further divergences from the reference (found via external review, then
# confirmed against github.com/state-spaces/mamba's mamba_simple.py /
# mixer_seq_simple.py directly, 2026-09-29) that the dt_proj fix alone did not
# address:
#
# 1. Reference Delta is NOT a direct d_inner->d_inner projection. x_proj emits a
#    combined (dt_rank + 2*d_state)-wide output; dt_proj is dt_rank->d_inner, where
#    dt_rank = ceil(d_model/16). This low-rank bottleneck changes both the parameter
#    count and, more importantly, the WEIGHT init scale: dt_init_std = dt_rank**-0.5,
#    not d_inner**-0.5. Since dt_rank << d_inner, this is a substantially larger
#    per-weight variance than a naive full-width projection would get.
# 2. Reference _init_weights() does NOT reinitialise Linear.weight at all (Embedding
#    excepted) -- it only zeros non-_no_reinit biases, plus a GPT-2-style 1/sqrt(N)
#    depth rescaling of residual OUTPUT projections (out_proj.weight) specifically.
#    Our earlier blanket normal(0, 0.02) on every Linear.weight was never reference
#    behaviour; it's also why dt_proj needed a weight-level _no_reinit hack that the
#    reference doesn't need (it only marks dt_proj.bias).
DT_RANK_DIVISOR = 16   # dt_rank = ceil(d_model / DT_RANK_DIVISOR), reference default

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


def selective_scan(delta: torch.Tensor, A: torch.Tensor, B_ssm: torch.Tensor,
                   C_ssm: torch.Tensor, u: torch.Tensor, scan: str = "sequential"
                   ) -> torch.Tensor:
    """Core selective-scan recurrence: h_t = A_bar_t h_{t-1} + B_bar_t u_t, y_t = C_t h_t.

    Factored out of MambaSSM.forward so it has exactly one implementation, callable
    directly with raw tensors -- this is what tests/test_selective_scan_parity.py
    compares against the vendored official `selective_scan_ref`
    (tests/vendor/selective_scan_ref.py) for forward and gradient agreement.

    Shapes: delta (B,T,E) positive (already softplus'd), A (E,N) negative real,
    B_ssm/C_ssm (B,T,N) -- shared across the E axis, matching MambaSSM's B/C
    projection -- u (B,T,E) is the scan input (h_conv in MambaSSM.forward).

    Returns y: (B,T,E), BEFORE the D skip connection and z gating (those are
    applied by the caller, matching the reference's `out = y + D*u; out *= silu(z)`
    ordering).
    """
    Bsz, T, E = u.shape
    if scan == "parallel":
        a = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        b = (delta.unsqueeze(-1) * B_ssm.unsqueeze(2)) * u.unsqueeze(-1)
        h = _associative_scan(a, b)                             # (B, T, E, N)
        return (h * C_ssm.unsqueeze(2)).sum(-1)                 # (B, T, E)

    state = torch.zeros(Bsz, E, A.shape[1], device=u.device, dtype=u.dtype)
    ys = []
    for t in range(T):
        dt = delta[:, t, :]                     # (B, E)
        b_t = B_ssm[:, t, :]                    # (B, N)
        c_t = C_ssm[:, t, :]                    # (B, N)
        x_t = u[:, t, :]                        # (B, E)
        A_bar = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))   # (B, E, N)
        B_bar = dt.unsqueeze(-1) * b_t.unsqueeze(1)             # (B, E, N)
        state = A_bar * state + B_bar * x_t.unsqueeze(-1)       # (B, E, N)
        y_t = (state * c_t.unsqueeze(1)).sum(-1)                # (B, E)
        ys.append(y_t)
    return torch.stack(ys, dim=1)               # (B, T, E)


class MambaSSM(nn.Module):
    """
    The selective SSM mixer at the core of each Mamba block.

    Shapes throughout (B=batch, T=seq_len, E=d_inner, N=d_state, R=dt_rank):
        in_proj  : (B, T, d_model) → (B, T, 2*E)     — x and gate z
        conv1d   : (B, E, T)       → (B, E, T)        — causal depthwise conv
        x_proj   : (B, T, E)       → (B, T, R+2*N)    — delta (low-rank), B, C
        dt_proj  : (B, T, R)       → (B, T, E)        — delta, pre-softplus (see DT_MIN/DT_MAX)
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
        self.dt_rank = math.ceil(d_model / DT_RANK_DIVISOR)

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

        # Project x → (delta_low, B, C) combined, matching the reference: delta goes
        # through a low-rank bottleneck (dt_rank) rather than a direct d_inner-wide
        # projection. bias=False for the whole thing -- B/C need no special init,
        # and delta's bias lives on dt_proj below, not here.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # Low-rank -> full-width delta projection. Weight init variance is scaled by
        # dt_rank (not d_inner) to preserve variance through the bottleneck -- this
        # is a real magnitude difference, not just a shape difference, since
        # dt_rank << d_inner. Bias set so softplus(bias) alone is log-uniform in
        # [DT_MIN, DT_MAX], before any input-dependent contribution is added.
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(DT_MAX) - math.log(DT_MIN))
            + math.log(DT_MIN)
        )
        # inverse softplus: softplus(inv_softplus(dt)) == dt
        inv_softplus_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus_dt)
        # Only the BIAS needs protecting from Mamba._init_weights() -- that function
        # no longer touches Linear.weight at all (see its docstring), matching the
        # reference, which likewise marks only dt_proj.bias `_no_reinit`.
        self.dt_proj.bias._no_reinit = True

        # A: (d_inner, d_state) — stored as log so exp(A_log) > 0,
        # and we negate to get stable negative-real diagonal eigenvalues
        A_init = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A_init = A_init.expand(self.d_inner, -1).log()   # log of 1..d_state
        self.A_log = nn.Parameter(A_init)

        # D: skip-connection weight (one per inner channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # ── Ablation-only, time-invariant Delta/B/C (see forward's input_independent) ──
        # Learned but NEVER a function of input -- broadcast over batch and time.
        # Only used when input_independent=True; dt_proj/x_proj are bypassed entirely
        # in that path, so no input-dependent computation reaches delta/B/C at all.
        # Init: fixed_delta_pre gets the same inverse-softplus targeting as dt_proj's
        # bias, so this ablation starts from a comparable Delta range rather than
        # being handicapped by a poor init; fixed_B/fixed_C get a modest-scale normal
        # init, since there is no equivalent "reference" init for a constant B/C.
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(DT_MAX) - math.log(DT_MIN))
            + math.log(DT_MIN)
        )
        self.fixed_delta_pre = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.fixed_B = nn.Parameter(torch.randn(d_state) * 0.1)
        self.fixed_C = nn.Parameter(torch.randn(d_state) * 0.1)

    def forward(self, x: torch.Tensor, return_delta: bool = False,
                input_independent: bool = False):
        """
        Args:
            x:            (B, T, d_model)
            return_delta: if True, also return delta_t for visualisation
            input_independent: ABLATION. Replace delta/B/C with genuinely time-invariant
                LEARNED parameters (fixed_delta_pre/fixed_B/fixed_C) that never see
                the input at all -- dt_proj and x_proj are bypassed entirely in this
                path. Used as the control for the Selective Copying gate (see
                src/tasks/selective_copy.py): a working selective implementation
                must beat this ablation by a wide margin.

                An earlier version of this ablation averaged delta/B/C over the time
                axis instead. That was contaminated: the "fixed" value at an early
                position depended on the whole sequence, including future positions,
                so it wasn't actually input-independent (external review,
                2026-09-29). This version has no such leak -- same recurrence
                (u, A, D unchanged), only the source of delta/B/C differs:

                    input-dependent selective SSM  vs  time-invariant learned SSM

                which is closer in spirit to a non-selective S4D-style model, though
                A here is still per-channel diagonal rather than S4's structured
                initialisation -- not a full S4 reimplementation.

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
        if input_independent:
            # Ablation: delta/B/C come from learned constants, never from h_conv.
            # dt_proj/x_proj are not called at all -- no input-dependent computation
            # reaches delta/B/C on this path. u (h_conv), A, and D are unchanged.
            delta = F.softplus(self.fixed_delta_pre).unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            B_ssm = self.fixed_B.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            C_ssm = self.fixed_C.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        else:
            dBC = self.x_proj(h_conv)                    # (B, T, R+2*N)
            delta_low = dBC[:, :, : self.dt_rank]        # (B, T, R) -- low-rank, per DT_RANK_DIVISOR
            B_ssm = dBC[:, :, self.dt_rank : self.dt_rank + self.d_state]      # (B, T, N)
            C_ssm = dBC[:, :, self.dt_rank + self.d_state :]                   # (B, T, N)

            delta = F.softplus(self.dt_proj(delta_low))  # (B, T, E); see DT_MIN/DT_MAX above

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
        y = selective_scan(delta, A, B_ssm, C_ssm, h_conv, scan=self.scan)

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

        self.n_layers = n_layers
        self._init_weights()

    def _init_weights(self):
        """Matches the reference model-level init exactly (see DT_RANK_DIVISOR
        docstring in this file for how this was found to differ from what we had).

        Reference behaviour, reproduced here:
          - nn.Linear biases are zeroed, UNLESS marked `_no_reinit` (dt_proj.bias).
            Conv1d bias is untouched -- the reference's check is `isinstance(m,
            nn.Linear)`, which a name-suffix rule like `name.endswith(".bias")`
            would get wrong for conv1d.bias.
          - nn.Embedding weight: normal(0, 0.02).
          - Every other Linear.weight (in_proj, x_proj, dt_proj, ...) is left at
            whatever it was after construction -- PyTorch's own default init, or a
            module's own deliberate init (dt_proj.weight is already set in
            MambaSSM.__init__ and is no longer at risk of being overwritten, since
            this function no longer touches Linear.weight in general).
          - EXCEPT `out_proj.weight`: the GPT-2 / Megatron residual-depth scheme
            (kaiming_uniform_, then divided by sqrt(n_layers)) -- each block's
            ssm.out_proj is one residual write, matching n_residuals_per_layer=1
            in the reference (we have no per-block MLP, same as their case).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None and not getattr(m.bias, "_no_reinit", False):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight"):
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(self.n_layers)

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
