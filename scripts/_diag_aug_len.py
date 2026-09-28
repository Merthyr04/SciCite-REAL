"""Verify the target-truncation fix caps every augmented sequence at max_length."""
from __future__ import annotations

import pickle
from pathlib import Path

from src.training.utils import get_tokenizer, tokenize_augmented

CACHE = Path("data/processed/cache/augment_f8696d53c362.pkl")

with open(CACHE, "rb") as fh:
    data = pickle.load(fh)

tok = get_tokenizer("bert")

for split in data:
    tr = data[split]
    cols = list(tr.keys())
    n = len(tr[cols[0]])
    worst = 0
    over = 0
    for i in range(n):
        total = len(
            tokenize_augmented(tok, tr["target"][i], list(tr["evidence"][i]), 256)["input_ids"]
        )
        worst = max(worst, total)
        if total > 256:
            over += 1
    print(f"{split:11s} n={n:5d} over_256={over} max_total={worst}")

# also check ACL-ARC augmented rows for the same failure mode
for tag in ("aclarc_bm25l",):
    caches = sorted(Path("data/processed/cache").glob("augment_*.pkl"))
    print("\ncache files:", [c.name for c in caches])