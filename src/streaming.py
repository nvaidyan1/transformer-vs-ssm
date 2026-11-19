"""
Token-by-token generation state for each architecture (plan2.md 2.2).

Each stepper consumes one token at a time and keeps exactly the tensors it must carry
to the next token; `state_bytes()` sums them. The tests in tests/test_streaming.py check
that stepping reproduces the model's full-sequence forward logits, so what is measured
is the real carried state, not an estimate.

  TransformerStepper  KV cache: per layer, keys and values for every past token.
  TCNStepper          per layer, the last (kernel_size - 1) * dilation normalised
                      inputs to that layer's conv (fewer while context is shorter).
  MambaStepper        per layer, the last (d_conv - 1) conv inputs and the SSM state
                      (d_inner x d_state). Custom model (src/models/mamba.py).

`analytic_bytes(...)` gives the same quantities from the shapes alone. Batch size 1,
float32 (4 bytes) throughout.
"""

import torch
import torch.nn.functional as F

BYTES = 4  # float32


def _nbytes(tensors) -> int:
    return sum(t.nelement() * t.element_size() for t in tensors)


class TransformerStepper:
    def __init__(self, model):
        self.m = model.eval()
        self.k = [None] * len(model.blocks)
        self.v = [None] * len(model.blocks)
        self.t = 0

    @torch.no_grad()
    def step(self, tok: torch.Tensor) -> torch.Tensor:
        """tok: (B,) LongTensor -> logits (B, vocab)."""
        m = self.m
        h = m.tok_emb(tok) + m.pos_emb.weight[self.t]
        for i, blk in enumerate(m.blocks):
            a = blk.attn
            x = blk.ln1(h)
            q, k, v = a.qkv(x).split(x.shape[-1], dim=-1)
            B = x.shape[0]
            q, k, v = (t.view(B, a.n_heads, 1, a.d_head) for t in (q, k, v))
            self.k[i] = k if self.k[i] is None else torch.cat([self.k[i], k], dim=2)
            self.v[i] = v if self.v[i] is None else torch.cat([self.v[i], v], dim=2)
            w = F.softmax((q @ self.k[i].transpose(-2, -1)) * a.scale, dim=-1)
            out = (w @ self.v[i]).transpose(1, 2).reshape(B, -1)
            h = h + a.out_proj(out)
            h = h + blk.ff(blk.ln2(h))
        self.t += 1
        return m.head(m.ln_f(h))

    def state_bytes(self) -> int:
        return _nbytes([t for t in self.k + self.v if t is not None])


class TCNStepper:
    def __init__(self, model):
        self.m = model.eval()
        self.buf = [None] * len(model.blocks)   # (B, d, <= pad) past normalised inputs

    @torch.no_grad()
    def step(self, tok: torch.Tensor) -> torch.Tensor:
        m = self.m
        x = m.tok_emb(tok)                                        # (B, d)
        for i, blk in enumerate(m.blocks):
            pad = blk.conv.padding
            hn = blk.norm(x).unsqueeze(-1)                        # (B, d, 1)
            past = self.buf[i]
            window = hn if past is None else torch.cat([past, hn], dim=-1)
            if window.shape[-1] < pad + 1:                        # causal zero padding
                window = F.pad(window, (pad + 1 - window.shape[-1], 0))
            h = blk.act(blk.conv.conv(window))[..., -1]           # (B, d)
            x = h + blk.res_conv(x.unsqueeze(-1))[..., 0]
            kept = window[..., -pad:] if pad > 0 else window[..., :0]
            n_real = min(pad, (0 if past is None else past.shape[-1]) + 1)
            self.buf[i] = kept[..., kept.shape[-1] - n_real:] if n_real else kept
        return m.head(m.ln_f(x))

    def state_bytes(self) -> int:
        return _nbytes([b for b in self.buf if b is not None])


class MambaStepper:
    def __init__(self, model):
        self.m = model.eval()
        n = len(model.blocks)
        self.conv = [None] * n      # (B, E, <= d_conv - 1) past conv inputs
        self.h = [None] * n         # (B, E, N) SSM state

    @torch.no_grad()
    def step(self, tok: torch.Tensor) -> torch.Tensor:
        m = self.m
        x = m.tok_emb(tok)                                        # (B, d_model)
        for i, blk in enumerate(m.blocks):
            s = blk.ssm
            x_in, z = s.in_proj(blk.norm(x)).chunk(2, dim=-1)     # (B, E) each
            k = s.d_conv
            past = self.conv[i]
            window = x_in.unsqueeze(-1) if past is None else torch.cat([past, x_in.unsqueeze(-1)], -1)
            full = F.pad(window, (k - window.shape[-1], 0))
            u = F.silu((full * s.conv1d.weight[:, 0, :]).sum(-1) + s.conv1d.bias)  # (B, E)
            self.conv[i] = window[..., -(k - 1):] if k > 1 else window[..., :0]
            dBC = s.x_proj(u)
            dlow = dBC[:, : s.dt_rank]
            Bm = dBC[:, s.dt_rank: s.dt_rank + s.d_state]
            Cm = dBC[:, s.dt_rank + s.d_state:]
            delta = F.softplus(s.dt_proj(dlow))                   # (B, E)
            A = -torch.exp(s.A_log)                               # (E, N)
            h = self.h[i]
            if h is None:
                h = torch.zeros(u.shape[0], s.d_inner, s.d_state, dtype=u.dtype)
            h = torch.exp(delta.unsqueeze(-1) * A) * h + \
                delta.unsqueeze(-1) * Bm.unsqueeze(1) * u.unsqueeze(-1)
            self.h[i] = h
            y = (h * Cm.unsqueeze(1)).sum(-1) + s.D * u
            x = x + s.out_proj(y * F.silu(z))
        return m.head(m.ln_f(x))

    def state_bytes(self) -> int:
        return _nbytes([t for t in self.conv + self.h if t is not None])


def analytic_bytes(arch: str, model, context: int, batch: int = 1) -> int:
    """Carried state after `context` tokens, from shapes only."""
    if arch == "transformer":
        d = model.tok_emb.weight.shape[1]
        return 2 * len(model.blocks) * context * d * batch * BYTES
    if arch == "tcn":
        d = model.tok_emb.weight.shape[1]
        return sum(min(context, b.conv.padding) for b in model.blocks) * d * batch * BYTES
    if arch == "mamba":
        total = 0
        for b in model.blocks:
            s = b.ssm
            total += s.d_inner * (min(context, s.d_conv - 1) + s.d_state)
        return total * batch * BYTES
    raise ValueError(arch)
