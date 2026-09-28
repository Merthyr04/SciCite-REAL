"""Measure how much of the target survives truncation on the CLS retrieval path.

The CLS path builds ``text = " ".join(evidence + [target])`` and tokenises it
with ``truncation=True, max_length=256``, which cuts from the right.  Because
the evidence sits in front of the target, long evidence can push the target out
of the window entirely.  This script quantifies that.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, r"e:\Paper\SciCite-REAL")

from src.training.utils import get_tokenizer, tokenize_augmented

MAX_LEN = 256
CACHE_DIR = Path(r"e:\Paper\SciCite-REAL\data\processed\cache")

tok = get_tokenizer("deberta")
sep = tok.sep_token_id
cls_ = tok.cls_token_id


def target_token_count(target: str) -> int:
    return len(tok(target, add_special_tokens=False)["input_ids"])


for path in sorted(CACHE_DIR.glob("augment_*.pkl")):
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    splits = list(data.keys())
    cols = list(data[splits[0]].keys())
    print(f"\n===== {path.name}  splits={splits}")
    print(f"      columns={cols}")
    for split in splits:
        tr = data[split]
        n = len(tr[cols[0]])
        if "evidence" not in tr:
            print(f"  {split:11s} n={n:5d}  (no evidence column)")
            continue
        full_gone = 0
        partial = 0
        ev_empty = 0
        # dual path: how many sequences exceed the window after the fix
        dual_over = 0
        dual_target_kept = 0
        for i in range(n):
            target = tr["target"][i]
            ev = list(tr["evidence"][i])
            if not ev:
                ev_empty += 1
            text = " ".join(ev + [target])
            ids = tok(text, truncation=True, max_length=MAX_LEN)["input_ids"]
            n_target = target_token_count(target)
            # target tokens occupy the tail of the untruncated sequence
            n_ev_prefix = len(tok(" ".join(ev), add_special_tokens=False)["input_ids"])
            kept = max(0, len(ids) - 1 - n_ev_prefix)  # minus [CLS]
            kept = min(kept, n_target)
            if kept == 0:
                full_gone += 1
            elif kept < n_target:
                partial += 1

            feats = tokenize_augmented(tok, target, ev, MAX_LEN)["input_ids"]
            if len(feats) > MAX_LEN:
                dual_over += 1
            # in the dual path the target always occupies the tail before [SEP]
            seg_start = len(feats) - 1 - n_target
            if seg_start >= 1:
                dual_target_kept += 1
        pct = lambda x: 100.0 * x / n
        print(
            f"  {split:11s} n={n:5d} | CLS path: target fully cut={full_gone:5d} "
            f"({pct(full_gone):5.1f}%)  partially cut={partial:5d} ({pct(partial):5.1f}%)"
            f" | empty evidence={ev_empty:4d} | dual: over_256={dual_over} "
            f"target_intact={dual_target_kept} ({pct(dual_target_kept):.1f}%)"
        )