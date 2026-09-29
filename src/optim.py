"""
Optimizer parameter grouping.

Why this module exists
----------------------
`AdamW(model.parameters(), weight_decay=0.1)` puts every parameter in one group and
decays all of them. That is not architecture-neutral, and in this repo it penalises
exactly the two models that lost the enwik8 comparison:

  - Mamba: `A_log` and `D` are the SSM's *dynamics*, not weights. Since
    `A = -exp(A_log)`, decaying `A_log` toward 0 drives `A` toward -1, distorting the
    state decay rate; `D` (skip connection) is pulled away from its init at 1.0. The
    reference Mamba implementation tags both `_no_weight_decay`.
    Affected here: 28 tensors / 121,856 params.
  - TCN: `weight_g` are weight-norm magnitude gains. Decaying them shrinks the
    effective gain of every conv layer; weight decay against weight norm is a known
    bad interaction. Affected here: 36 tensors / 9,216 params.

Rule applied
------------
No weight decay for:
  - SSM dynamics      (`*.A_log`, `*.D`)
  - weight-norm gains (`*weight_g`)
  - anything with ndim < 2 (LayerNorm gains, biases)
Weight decay for everything else (2-D+ weights, including tied embeddings — present
identically in all three architectures, so decaying them is symmetric and does not
bias the comparison).
"""

import torch

NO_DECAY_SUFFIXES = ("A_log", ".D")
NO_DECAY_SUBSTRINGS = ("weight_g",)


def is_no_decay(name: str, param: torch.nn.Parameter) -> bool:
    """True if this parameter should be exempt from weight decay."""
    if any(name.endswith(s) for s in NO_DECAY_SUFFIXES):
        return True
    if any(s in name for s in NO_DECAY_SUBSTRINGS):
        return True
    return param.ndim < 2


def build_param_groups(model: torch.nn.Module, weight_decay: float) -> list:
    """Split model parameters into decay / no-decay groups.

    Returns a list of two param-group dicts suitable for torch.optim.AdamW.
    Every trainable parameter appears in exactly one group.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if is_no_decay(name, p) else decay).append(p)

    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def describe_param_groups(model: torch.nn.Module) -> str:
    """Human-readable summary — printed at train start so runs are auditable."""
    d_t = d_p = n_t = n_p = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if is_no_decay(name, p):
            n_t += 1
            n_p += p.numel()
        else:
            d_t += 1
            d_p += p.numel()
    return (f"  param groups: decay {d_t} tensors / {d_p:,} params | "
            f"no-decay {n_t} tensors / {n_p:,} params")
