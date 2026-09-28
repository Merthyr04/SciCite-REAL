"""Compare the evidence retrieved by BM25L vs BM25 on the same SciCite pool.

The manuscript claims the two scorers ``rank the pool almost identically``; if
the retrieved top-3 sets differ substantially, that claim is wrong and the
length-normalisation difference is doing real work.
"""
from __future__ import annotations

import pickle
from pathlib import Path

CACHE = Path(r"e:\Paper\SciCite-REAL\data\processed\cache")

BM25L = CACHE / "augment_f8696d53c362.pkl"  # scicite bm25l, leak0
BM25 = CACHE / "augment_4130abef2f4a.pkl"  # scicite bm25, leak0

with open(BM25L, "rb") as fh:
    a = pickle.load(fh)
with open(BM25, "rb") as fh:
    b = pickle.load(fh)

for split in ("train", "validation", "test"):
    ea = a[split]["evidence"]
    eb = b[split]["evidence"]
    n = len(ea)
    identical = 0
    overlap = 0
    total = 0
    mean_len_a = 0
    mean_len_b = 0
    for i in range(n):
        sa, sb = list(ea[i]), list(eb[i])
        if sa == sb:
            identical += 1
        overlap += len(set(sa) & set(sb))
        total += max(len(sa), len(sb))
        mean_len_a += sum(len(x) for x in sa) / max(1, len(sa))
        mean_len_b += sum(len(x) for x in sb) / max(1, len(sb))
    print(
        f"{split:11s} n={n:5d}  identical_topk={identical:5d} ({100*identical/n:5.1f}%)  "
        f"set_overlap={100*overlap/total:5.1f}%  "
        f"mean_evidence_chars bm25l={mean_len_a/n:6.1f} bm25={mean_len_b/n:6.1f}"
    )