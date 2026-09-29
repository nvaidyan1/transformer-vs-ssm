"""
Print an MQAR example in human-readable form. Use this to eyeball the task before
training anything on it.

    python scripts/inspect_mqar.py
    python scripts/inspect_mqar.py --seq_len 48 --n_pairs 3 --vocab_size 41 --seed 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks.mqar import generate_mqar, min_seq_len, vocab_layout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--n_pairs", type=int, default=3)
    ap.add_argument("--n_queries", type=int, default=None)
    ap.add_argument("--vocab_size", type=int, default=41)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--constant_filler", action="store_true",
                    help="Use the debugging filler (all id 0) instead of random filler.")
    args = ap.parse_args()

    lay = vocab_layout(args.vocab_size)
    s = generate_mqar(args.seq_len, args.n_pairs, args.vocab_size,
                      n_queries=args.n_queries, seed=args.seed,
                      random_filler=not args.constant_filler)

    q_at = {q["query_position"]: q for q in s.meta["queries"]}
    src_at = {q["source_position"]: q for q in s.meta["queries"]}

    print(f"seq_len={args.seq_len}  n_pairs={args.n_pairs}  "
          f"n_queries={s.meta['n_queries']}  vocab={args.vocab_size}  seed={args.seed}")
    print(f"keys ids [{lay['key_lo']},{lay['key_hi']})  "
          f"values ids [{lay['val_lo']},{lay['val_hi']})  "
          f"min_seq_len={min_seq_len(args.n_pairs, s.meta['n_queries'])}")
    print()

    # Role annotation per position
    roles = []
    for i, tok in enumerate(s.tokens):
        if i in src_at:
            roles.append("K")
        elif i - 1 in src_at:
            roles.append("v")
        elif i in q_at:
            roles.append("Q")
        elif i - 1 in q_at:
            roles.append("a")
        else:
            roles.append(".")

    per_line = 16
    for start in range(0, args.seq_len, per_line):
        idx = range(start, min(start + per_line, args.seq_len))
        print("  pos  " + " ".join(f"{i:>4d}" for i in idx))
        print("  tok  " + " ".join(f"{s.tokens[i]:>4d}" for i in idx))
        print("  role " + " ".join(f"{roles[i]:>4s}" for i in idx))
        print("  tgt  " + " ".join(("    ." if s.targets[i] < 0 else f"{s.targets[i]:>4d}")
                                   for i in idx))
        print()

    print("legend: K=key presented  v=its value  Q=query  a=answer(teacher-forced)  .=filler")
    print("        loss is taken at Q positions only\n")
    print("queries:")
    for q in s.meta["queries"]:
        print(f"  key {q['key']:>3d} -> value {q['value']:>3d} | "
              f"presented @{q['source_position']:>3d}  queried @{q['query_position']:>3d}  "
              f"distance {q['retrieval_distance']:>3d}")

    occ = int((s.tokens == s.meta["queries"][0]["key"]).sum())
    print(f"\nsanity: first queried key occurs {occ} times in the sequence (expect 2)")
    print(f"        loss_mask selects {int(s.loss_mask.sum())} positions "
          f"(expect {s.meta['n_queries']})")


if __name__ == "__main__":
    main()
