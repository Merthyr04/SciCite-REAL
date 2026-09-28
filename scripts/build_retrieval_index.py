#!/usr/bin/env python3
"""Build & cache the BM25 retrieval index over the training corpus.

The retrieval pool must be label-safe: for evaluation we must never retrieve
from the test split. This script default index is built over the *train*
contexts, so index.path is safe to use for both train and test retrieval.

You may optionally exclude same-label neighbours (leak-free ablation) — the flag
is consumed at retrieval time, not here.

Usage:
    python scripts/build_retrieval_index.py \
        --data data/processed/scicite \
        --out data/index/scicite_bm25l.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from src.data.dataset import read_jsonl
from src.retrieval.index import BM25Index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/scicite", help="preprocessed dir")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--out", default="data/index/scicite_bm25l_train.pkl")
    ap.add_argument("--variant", default="bm25l", choices=["bm25", "bm25l"])
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--delta", type=float, default=0.5)
    args = ap.parse_args()

    rows = read_jsonl(Path(args.data) / f"{args.split}.jsonl")
    texts = [r["string"] for r in rows]
    labels = [r["label"] for r in rows]

    index = BM25Index(texts, variant=args.variant, k1=args.k1, b=args.b, delta=args.delta)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index.corpus = texts
    index.labels = labels  # store alongside so leak-free filtering can use it
    with open(out_path, "wb") as fh:
        pickle.dump(index, fh)
    print(f"[index] variant={args.variant} docs={len(texts)} -> {out_path}")


if __name__ == "__main__":
    main()