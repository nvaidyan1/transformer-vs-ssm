"""
Per-parameter std at step 0, custom vs official Mamba, side by side (plan2.md 0.2
amendment, item 1). Init only; no training.

Uses the real mamba-ssm mixer when importable, otherwise the vendored copy of its
constructor (tests/vendor/mamba_simple_init.py, identical init code).

Usage:
    python scripts/init_std_compare.py --out results/phase0_weight_copy/init_std.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from src.models import build_model
from src.models.mamba_official import OracleMamba

FLAG_REL = 0.10


def mixer_cls():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        return Mamba, "mamba_ssm"
    except Exception:
        from vendor.mamba_simple_init import Mamba
        return Mamba, "vendored"


def compare(emb_init_std, model_cfg, vocab_size, seed=0):
    cls, _ = mixer_cls()
    cfg = {**model_cfg, "emb_init_std": emb_init_std, "out_proj_depth_scale": True}
    torch.manual_seed(seed)
    custom = build_model("mamba", vocab_size=vocab_size, **cfg)
    torch.manual_seed(seed)
    official = OracleMamba(vocab_size=vocab_size, mixer_cls=cls, **cfg)
    off = {n.replace(".mixer.", ".ssm."): p for n, p in official.named_parameters()}
    rows = []
    for n, p in custom.named_parameters():
        if ".fixed_" in n:
            continue
        a, b = p.std().item(), off[n].std().item()
        ratio = a / b if b > 0 else (1.0 if a == 0 else float("inf"))
        rows.append({"param": n, "custom_std": a, "official_std": b, "ratio": ratio,
                     "flag": abs(ratio - 1) > FLAG_REL})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/recall/mamba_small.yaml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    _, source = mixer_cls()
    out = {"official_mixer": source, "out_proj_depth_scale": True, "by_emb": {}}
    for emb in (0.02, 1.0):
        rows = compare(emb, cfg["model"], cfg["vocab_size"])
        out["by_emb"][str(emb)] = rows
        print(f"\n== emb_init_std={emb}  (official mixer: {source})")
        print(f"{'param':32s} {'custom std':>11s} {'official std':>12s} {'ratio':>7s}")
        for r in rows:
            print(f"{r['param']:32s} {r['custom_std']:11.5f} {r['official_std']:12.5f} "
                  f"{r['ratio']:7.3f}{'  <-- >10%' if r['flag'] else ''}")
        flagged = [r["param"] for r in rows if r["flag"]]
        print(f"differ by >10%: {flagged or 'none'} (of {len(rows)})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
