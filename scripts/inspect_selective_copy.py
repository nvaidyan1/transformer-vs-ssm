"""
Print a Selective Copying example in human-readable form.

    python scripts/inspect_selective_copy.py
    python scripts/inspect_selective_copy.py --seq_len 40 --n_data 4 --vocab_size 12 --seed 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tasks.selective_copy import (MARKER_ID, NOISE_ID, generate_selective_copy,
                                      min_seq_len)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq_len", type=int, default=40)
    ap.add_argument("--n_data", type=int, default=4)
    ap.add_argument("--vocab_size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_examples", type=int, default=2,
                    help="Show several, to make the randomised spacing visible.")
    args = ap.parse_args()

    print(f"seq_len={args.seq_len}  n_data={args.n_data}  vocab={args.vocab_size}  "
          f"min_seq_len={min_seq_len(args.n_data)}")
    print("legend: D=data  .=noise  M=marker  o=output(teacher-forced)\n")

    for ex in range(args.n_examples):
        s = generate_selective_copy(args.seq_len, args.n_data, args.vocab_size,
                                    seed=args.seed + ex)
        m = s.meta["marker_position"]
        src = set(s.meta["source_positions"])

        roles = []
        for i in range(args.seq_len):
            if i in src:
                roles.append("D")
            elif i == m:
                roles.append("M")
            elif i > m:
                roles.append("o")
            else:
                roles.append(".")

        per_line = 20
        print(f"--- example seed={args.seed + ex}  gaps={s.meta['gaps']} ---")
        for start in range(0, args.seq_len, per_line):
            idx = range(start, min(start + per_line, args.seq_len))
            print("  pos  " + " ".join(f"{i:>3d}" for i in idx))
            print("  tok  " + " ".join(f"{s.tokens[i]:>3d}" for i in idx))
            print("  role " + " ".join(f"{roles[i]:>3s}" for i in idx))
            print("  tgt  " + " ".join(("  ." if s.targets[i] < 0 else f"{s.targets[i]:>3d}")
                                       for i in idx))
        print(f"  data {s.meta['data']} @ {s.meta['source_positions']}  "
              f"-> reproduced at {list(range(m, m + args.n_data))}")
        print()

    print("Note the source positions move between examples: that randomised spacing is")
    print("what prevents a fixed-offset solution, and what makes this a gate on")
    print("selectivity rather than another benchmark number.")


if __name__ == "__main__":
    main()
