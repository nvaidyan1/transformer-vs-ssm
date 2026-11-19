"""
plan2.md 5.3: effective gain vs Kalman gain, and Delta by noise regime, on held-out
switching-noise sequences, for every saved switching model (selective Mamba, ablation,
GRU; 1 and 2 layers; seeds 0-2). Writes one JSON with per-model results and the data for
one example trace.

Usage:
    python scripts/phase5_gain.py --models results/phase5/models --out results/phase5/gain.json
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.filter_analysis import delta_by_regime, effective_gain, pearson
from src.models.regression import build_regression
from src.tasks.filtering import generate, kalman
from src.train_task import write_record

HELDOUT_SEED = 3_000_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    u, x, r, flag = generate("switching", args.batch, args.seq_len, HELDOUT_SEED)
    kal_est, K = kalman(u[..., 0].astype(float), r)
    ut = torch.from_numpy(u).to(dev)
    rec = {"heldout_seed": HELDOUT_SEED, "batch": args.batch, "seq_len": args.seq_len,
           "kalman_gain_low": float(K[flag == 0].mean()), "kalman_gain_high": float(K[flag == 1].mean()),
           "models": {}}
    example = {"y": u[0, :, 0].tolist(), "x": x[0].tolist(), "flag": flag[0].tolist(),
               "kalman": kal_est[0].tolist(), "kalman_gain": K[0].tolist(), "estimates": {}}
    for p in sorted(glob.glob(str(Path(args.models) / "switching_*.pt"))):
        ck = torch.load(p, map_location="cpu", weights_only=True)
        m = build_regression(ck["arch"], ck["in_dim"], ck["d_model"], ck["n_layers"],
                             scan=ck.get("scan", "parallel"))
        m.load_state_dict(ck["state_dict"]); m.to(dev).eval()
        kw = {"input_independent": True} if ck["arch"] == "mamba_ablated" else {}
        g = effective_gain(m, ut, **kw)
        name = Path(p).stem
        res = {"arch": ck["arch"], "n_layers": ck["n_layers"], "pearson_r": pearson(g, K),
               "gain_low": float(g[flag == 0].mean()), "gain_high": float(g[flag == 1].mean()),
               "gain_sample": g[:4].tolist()}
        if ck["arch"] == "mamba":
            res["delta"] = delta_by_regime(m, ut, flag)
        with torch.no_grad():
            est = m(ut, **kw)[0].cpu().numpy()
        example["estimates"][name] = est.tolist()
        rec["models"][name] = res
        print(f"{name}: r={res['pearson_r']:.3f} gain low {res['gain_low']:.3f} "
              f"high {res['gain_high']:.3f}", flush=True)
    rec["kalman_gain_sample"] = K[:4].tolist()
    rec["example"] = example
    write_record(rec, args.out)


if __name__ == "__main__":
    main()
