#!/usr/bin/env python3
"""Train a citation-intent classifier (baseline or BM25-augmented).

This is the main entry point that reproduces every table row in the paper:

  Baseline (no retrieval):
      python scripts/run_training.py --model bert --tag baseline_bert
      python scripts/run_training.py --model scibert --tag baseline_scibert
      python scripts/run_training.py --model deberta --tag baseline_deberta

  Retrieval-augmented (BM25 evidence prepended):
      python scripts/run_training.py --model deberta --retrieval \
          --index data/index/scicite_bm25l_train.pkl --top-k 3 \
          --tag real_deberta_k3

  Ablations (leak-free, sequence length, top-k, no-index):
      python scripts/run_training.py --model deberta --retrieval --leak-free \
          --tag ablation_real_deberta_leakfree
      python scripts/run_training.py --model deberta --retrieval --max-length 128 --top-k 1 \
          --tag ablation_real_deberta_len128_k1
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import load_scicite, PRIMARY_LABELS, read_jsonl
from src.retrieval.retrieval_augmented import RetrievalAugmenter
from src.retrieval.index import BM25Index
from src.training.trainer import train
from src.training.utils import get_tokenizer, set_seed, tokenize_function


def build_augmented_splits(dd, index_path: str, top_k: int, leak_free: bool):
    """Prepend BM25 evidence to every instance's input text."""
    from datasets import Dataset

    index = None
    if index_path:
        with open(index_path, "rb") as fh:
            index = pickle.load(fh)

    augmenter = RetrievalAugmenter(
        index=index,
        top_k=top_k,
        evidence_source="train",
        leak_free=leak_free,
        labels=PRIMARY_LABELS,
    )

    new_splits = {}
    for split in dd:
        rows = []  # tokenise after augmentation
        for inst in dd[split]:
            aug = augmenter.augment(
                target_text=inst["string"],
                target_label=inst["label"],
                doc_id=inst["id"],
            )
            rows.append(
                {
                    "text": aug["text"],
                    "target": aug["target"],
                    "evidence": aug["evidence"],
                    "label_id": inst["label_id"],
                    "label": inst["label"],
                }
            )
        new_splits[split] = Dataset.from_list(rows)
    return new_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/scicite")
    ap.add_argument("--model", default="deberta",
                    choices=["bert", "scibert", "deberta", "deberta-small"])
    ap.add_argument("--tag", default="exp", help="run id used for output dir")
    ap.add_argument("--output-root", default="runs")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accumulation", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freeze-embeddings", action="store_true")
    ap.add_argument("--no-fp16-inference-check", action="store_true", help="unused; kept for CLI parity")
    ap.add_argument("--no-mixed-precision", action="store_true", help="train in pure fp32 (diagnostic)")
    ap.add_argument("--no-grad-checkpointing", action="store_true", help="disable gradient checkpointing (diagnostic)")
    ap.add_argument("--retrieval", action="store_true",
                    help="enable BM25 retrieval augmentation")
    ap.add_argument("--index", default="data/index/scicite_bm25l_train.pkl")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--leak-free", action="store_true",
                    help="forbid retrieving contexts with the same label")
    ap.add_argument("--pooling", default="cls", choices=["cls", "dual"],
                    help="'dual' enables the evidence-aware gated fusion head")
    ap.add_argument("--preserve-target", action="store_true",
                    help="with --pooling cls, tokenise segment-aware so the target "
                         "survives truncation (isolates truncation policy from head)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'})")

    dd = load_scicite(args.data)

    if args.retrieval:
        from datasets import DatasetDict
        import hashlib

        # Deterministic per-config cache: exact-match augmentation is expensive
        # (O(n) BM25 scoring per instance in pure Python), so persist it on disk
        # and reuse across runs/ablations with the same (top_k, leak_free) pair.
        cache_dir = Path(args.data) / ".." / "cache"
        cache_key = hashlib.sha1(
            f"augment_topk{args.top_k}_leak{int(args.leak_free)}_src{args.index}".encode()
        ).hexdigest()[:12]
        cache_path = (cache_dir / f"augment_{cache_key}.pkl").resolve()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            print(f"[augment] loading cached evidence from {cache_path}")
            from datasets import Dataset
            with open(cache_path, "rb") as fh:
                dd = DatasetDict(
                    {k: Dataset.from_dict(v) for k, v in pickle.load(fh).items()}
                )
        else:
            new_splits = build_augmented_splits(
                dd, args.index, args.top_k, args.leak_free
            )
            dd = DatasetDict(new_splits)
            with open(cache_path, "wb") as fh:
                pickle.dump({k: v.to_dict() for k, v in new_splits.items()}, fh)
            print(f"[augment] cached evidence to {cache_path}")
        print(f"[augment] BM25 top_k={args.top_k} leak_free={args.leak_free}")
    else:
        # unify the text column so the trainer's single-input tokenizer works
        dd = dd.rename_column("string", "text")

    results = train(
        dataset=dd,
        model_name=args.model,
        max_length=args.max_length,
        per_device_batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,
        learning_rate=args.lr,
        epochs=args.epochs,
        dropout=args.dropout,
        freeze_embeddings=args.freeze_embeddings,
        pooling=args.pooling,
        target_preserving=args.preserve_target,
        use_fp16=not args.no_mixed_precision,
        use_gradient_checkpointing=not args.no_grad_checkpointing,
        output_dir=f"{args.output_root}/{args.tag}",
        seed=args.seed,
    )
    results.update(
        {
            "tag": args.tag,
            "model": args.model,
            "retrieval": args.retrieval,
            "top_k": args.top_k if args.retrieval else None,
            "leak_free": args.leak_free if args.retrieval else None,
            "max_length": args.max_length,
            "pooling": args.pooling,
            "preserve_target": args.preserve_target,
            "seed": args.seed,
        }
    )
    out_path = Path(args.output_root) / args.tag / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== RESULT ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()