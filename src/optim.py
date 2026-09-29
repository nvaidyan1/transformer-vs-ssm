"""
Optimizer parameter grouping.

Two independent axes, so every parameter lands in one of up to four groups:

  1. decay / no-decay (weight decay)
  2. base-lr / boosted-lr (learning rate multiplier)

Axis 1 -- why weight decay is grouped
--------------------------------------
`AdamW(model.parameters(), weight_decay=0.1)` puts every parameter in one group and
decays all of them. That is not architecture-neutral, and in this repo it penalises
exactly the two models that lost the enwik8 comparison:

  - Mamba: `A_log` and `D` are the SSM's *dynamics*, not weights. Since
    `A = -exp(A_log)`, decaying `A_log` toward 0 drives `A` toward -1, distorting the
    state decay rate; `D` (skip connection) is pulled away from its init at 1.0. The
    reference Mamba implementation tags both `_no_weight_decay`.
  - TCN: `weight_g` are weight-norm magnitude gains. Decaying them shrinks the
    effective gain of every conv layer; weight decay against weight norm is a known
    bad interaction.

No weight decay for: SSM dynamics (`*.A_log`, `*.D`), weight-norm gains (`*weight_g`),
and anything with ndim < 2 (LayerNorm gains, biases). Weight decay for everything else.

Axis 2 -- why learning rate is grouped
-----------------------------------------
Found via the Selective Copying gate (see PLAN.md, tests/test_mamba_dt_init.py): after
fixing Delta's initialisation, Mamba still could not learn the task at any single global
learning rate from 1e-4 to 1e-2 (tested locally; loss stayed at ln(vocab_size), i.e. no
learning at all). Instrumenting gradients explained why:

    param                        |grad| mean
    tok_emb.weight                  1.2e-02
    blocks.0.ssm.in_proj.weight     8.1e-04
    blocks.0.ssm.conv1d.weight      1.9e-04
    blocks.0.ssm.out_proj.weight    1.2e-03
    blocks.0.ssm.x_proj.weight      9.8e-06   <- 2-4 orders of magnitude smaller
    blocks.0.ssm.dt_proj.weight     2.5e-08   <-
    blocks.0.ssm.dt_proj.bias       1.9e-07   <-
    blocks.0.ssm.A_log              5.5e-08   <-

The selectivity parameters (dt_proj, x_proj, A_log) sit 4-5 orders of magnitude below
the rest of the network -- close enough to AdamW's eps (1e-8) that its per-parameter
normalisation damps rather than corrects for the scale gap. A single global LR cannot
close a gap that size: raising it 33x (3e-4 -> 1e-2) produced no measurable change.
This is a known characteristic of SSM training, not specific to this implementation --
reference recipes commonly give dt/A/selectivity parameters their own, larger, LR.

`ssm_lr_scale` multiplies the base LR for exactly these parameters. Default 1.0 (off),
so existing configs are unaffected unless a config or CLI flag sets it explicitly.
"""

import torch

NO_DECAY_SUFFIXES = ("A_log", ".D")
NO_DECAY_SUBSTRINGS = ("weight_g",)

# Parameters whose gradients operate on a different scale than the rest of the
# network (see module docstring). `x_proj` here means the B/C projection specifically
# (post-refactor it no longer carries delta) -- matched by suffix so it does not
# accidentally catch an unrelated "x_proj" in some other architecture.
SSM_SELECTIVITY_SUFFIXES = ("dt_proj.weight", "dt_proj.bias", "A_log", "x_proj.weight")


def is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    """True if this parameter should be exempt from weight decay."""
    if any(name.endswith(s) for s in NO_DECAY_SUFFIXES):
        return True
    if any(s in name for s in NO_DECAY_SUBSTRINGS):
        return True
    return param.ndim < 2


def is_ssm_selectivity(name: str) -> bool:
    """True if this parameter should receive the boosted learning-rate multiplier."""
    return any(name.endswith(s) for s in SSM_SELECTIVITY_SUFFIXES)


def build_param_groups(model: torch.nn.Module, weight_decay: float,
                       ssm_lr_scale: float = 1.0) -> list:
    """Split model parameters into up to four groups (decay x lr-scale).

    Each returned dict carries a `lr_scale` key (not a native AdamW field) that the
    training loop must multiply into the scheduled LR every step -- see
    src/train_task.py, which does `g['lr'] = base_lr * g['lr_scale']`. Passing these
    groups straight to AdamW without that multiplication silently drops the scaling
    the first time the LR schedule updates any group's 'lr'.

    Returns a list of param-group dicts suitable for torch.optim.AdamW. Every
    trainable parameter appears in exactly one group; empty groups are omitted.
    """
    buckets = {}  # (no_decay: bool, boosted: bool) -> [params]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (is_no_decay(name, p), is_ssm_selectivity(name))
        buckets.setdefault(key, []).append(p)

    groups = []
    for (no_decay, boosted), params in buckets.items():
        groups.append({
            "params": params,
            "weight_decay": 0.0 if no_decay else weight_decay,
            "lr_scale": ssm_lr_scale if boosted else 1.0,
        })
    return groups


def describe_param_groups(model: torch.nn.Module, ssm_lr_scale: float = 1.0) -> str:
    """Human-readable summary — printed at train start so runs are auditable."""
    counts = {}  # (no_decay, boosted) -> [n_tensors, n_params]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (is_no_decay(name, p), is_ssm_selectivity(name))
        c = counts.setdefault(key, [0, 0])
        c[0] += 1
        c[1] += p.numel()

    lines = ["  param groups:"]
    labels = {(False, False): "decay, base-lr",
              (True, False): "no-decay, base-lr",
              (False, True): f"decay, {ssm_lr_scale}x-lr",
              (True, True): f"no-decay, {ssm_lr_scale}x-lr"}
    for key, (t, p) in sorted(counts.items()):
        lines.append(f"    {labels[key]:22s}: {t:3d} tensors / {p:,} params")
    return "\n".join(lines)
