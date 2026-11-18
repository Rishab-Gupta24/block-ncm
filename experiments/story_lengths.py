"""Milestone 2: how many TinyStories documents are long enough for M chunks of 128 tokens?

Reads the offsets array shipped with hazyresearch/ncm-datasets (tinystories/train_bpe_8192_offsets.npy),
which marks document boundaries in the BPE-8192 token stream.

    python experiments/story_lengths.py ~/ncm/data/tinystories/train_bpe_8192_offsets.npy
"""
import sys

import numpy as np

CHUNK = 128

off = np.load(sys.argv[1])
print(f"offsets: shape={off.shape} dtype={off.dtype} first={off[:5].tolist()} last={off[-3:].tolist()}")
off = off.astype(np.int64).ravel()
if not np.all(np.diff(off) >= 0):
    sys.exit("offsets are not monotone; layout differs from assumption (cumulative start positions)")

lengths = np.diff(off)
n = len(lengths)
print(f"documents: {n:,}   total tokens: {lengths.sum():,}")
q = np.percentile(lengths, [5, 25, 50, 75, 95, 99])
print("length percentiles (5/25/50/75/95/99): " + " / ".join(f"{x:.0f}" for x in q))
print(f"mean {lengths.mean():.1f}   max {lengths.max()}")
print()
print(f"{'M':>3} {'min tokens':>11} {'docs':>12} {'share':>8}")
for M in (1, 2, 3, 4, 6, 8):
    k = int((lengths >= M * CHUNK).sum())
    print(f"{M:>3} {M * CHUNK:>11} {k:>12,} {k / n:>8.1%}")
