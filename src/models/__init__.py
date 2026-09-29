"""
Model registry. All three architectures share the same interface:

    model = build_model(arch, **overrides)
    logits = model(x)   # x: (batch, seq_len)  →  (batch, seq_len, 256)
"""

from src.models.transformer import Transformer
from src.models.tcn import TCN
from src.models.mamba import Mamba

# Default configs sized to ~10M parameters each.
# Exact counts documented after count_parameters() is called at init time.
_CONFIGS = {
    "transformer": dict(
        n_layers=11,     # → 9,790,720 params
        d_model=256,
        n_heads=4,
        d_ff=1024,
        dropout=0.1,
        max_seq_len=4096,
    ),
    "tcn": dict(
        n_layers=18,     # → 9,596,416 params
        d_model=256,
        kernel_size=7,
        dropout=0.1,
    ),
    "mamba": dict(
        n_layers=14,     # → 9,635,328 params
        d_model=256,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ),
}

_CLASSES = {
    "transformer": Transformer,
    "tcn": TCN,
    "mamba": Mamba,
}


def build_model(arch: str, vocab_size: int = 256, **overrides):
    """Instantiate a model by name with optional config overrides.

    Args:
        arch:        one of 'transformer', 'tcn', 'mamba'
        vocab_size:  token vocabulary (256 = raw bytes, the enwik8 setting).
                     Synthetic-task runs may use a smaller vocabulary; it must be
                     identical across architectures or the comparison is unfair.
        **overrides: any config key to override the default

    Returns:
        nn.Module with interface logits = model(x)
    """
    if arch not in _CLASSES:
        raise ValueError(f"Unknown arch {arch!r}. Choose from {list(_CLASSES)}")
    cfg = {**_CONFIGS[arch], **overrides}
    cfg.pop("vocab_size", None)
    return _CLASSES[arch](vocab_size=vocab_size, **cfg)


def count_parameters(model) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
