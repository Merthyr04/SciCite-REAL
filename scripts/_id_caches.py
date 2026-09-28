"""Identify which augmented cache file corresponds to which retrieval config."""
from __future__ import annotations

import hashlib
from pathlib import Path

CACHE = Path(r"e:\Paper\SciCite-REAL\data\processed\cache")

candidates = []
for idx in (
    "data/index/scicite_bm25l_train.pkl",
    "data/index/scicite_bm25_train.pkl",
    "data/index/aclarc_bm25l_train.pkl",
):
    for leak in (0, 1):
        for k in (3,):
            candidates.append((k, leak, idx))

print("present cache files:")
for p in sorted(CACHE.glob("augment_*.pkl")):
    print(f"  {p.name}")

print("\nrecomputed keys:")
for k, leak, idx in candidates:
    key = hashlib.sha1(f"augment_topk{k}_leak{leak}_src{idx}".encode()).hexdigest()
    hit = (CACHE / f"augment_{key}.pkl").exists()
    print(f"  topk={k} leak={leak} src={idx:38s} -> augment_{key}.pkl  {'<== MATCH' if hit else ''}")