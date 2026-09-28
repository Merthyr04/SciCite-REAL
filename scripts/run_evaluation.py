#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the test split and dump reports.

Produces ``{tag}/summary.json`` (Accuracy + Macro-F1) and, when requested, a
per-class table and confusion matrix.

Usage:
    python scripts/run_evaluation.py --checkpoint runs/baseline_deberta/best_model \
        --data data/processed/scicite --tag eval_baseline_deberta --report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.dataset import load_scicite, PRIMARY_LABELS, PRIMARY_LABEL2ID
from src.evaluation.metrics import (
    accuracy,
    macro_f1,
    classification_report_dict,
    confusion_matrix_df,
)
from src.models.classifier import build_model
from src.training.utils import tokenize_augmented, tokenize_function


def _pad_to_len(seqs):
    """Right-pad a ragged list of integer sequences to the common max length."""
    maxlen = max(len(s) for s in seqs)
    return [list(s) + [0] * (maxlen - len(s)) for s in seqs]


def load_eval_model(checkpoint, tokenizer, pooling, num_labels):
    """Load the appropriate head; the dual-pooling head is a custom module."""
    cap = Path(checkpoint) / "head_config.json"
    pooling = pooling or (json.loads(cap.read_text())["pooling"] if cap.exists() else "cls")
    if pooling == "dual":
        model = build_model("deberta", num_labels=num_labels, pooling="dual")
        sd = torch.load(Path(checkpoint) / "dual_head.pt", map_location="cpu",
                        weights_only=True)
        model.load_state_dict(sd)
        return model, pooling
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    return model, pooling


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/processed/scicite")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--report", action="store_true", help="write per-class + confusion matrix")
    ap.add_argument("--index", default="data/index/scicite_bm25l_train.pkl",
                    help="needed to rebuild evidence for dual-pooling eval")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--leak-free", action="store_true")
    args = ap.parse_args()

    dd = load_scicite(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    model, pooling = load_eval_model(
        args.checkpoint, tokenizer, pooling=None, num_labels=len(PRIMARY_LABELS)
    )
    use_dual = pooling == "dual"

    if use_dual:
        index = None
        if Path(args.index).exists():
            import pickle
            with open(args.index, "rb") as fh:
                index = pickle.load(fh)
        from src.retrieval.retrieval_augmented import RetrievalAugmenter
        aug = RetrievalAugmenter(index=index, top_k=args.top_k,
                                 leak_free=args.leak_free, labels=PRIMARY_LABELS)
        rows = []
        for ex in dd["test"]:
            a = aug.augment(ex["string"], ex["label"], doc_id=ex["id"])
            feats = tokenize_augmented(tokenizer, a["target"], a["evidence"], args.max_length)
            rows.append({**feats, "label_id": ex["label_id"]})
        from datasets import Dataset
        test = Dataset.from_list(rows)
    else:
        tok = tokenize_function(tokenizer, max_length=args.max_length)
        test = dd["test"].map(tok, batched=True, remove_columns=["string"])

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    preds, labels = [], []
    from torch.utils.data import DataLoader, TensorDataset

    input_ids = torch.tensor(_pad_to_len(test["input_ids"]))
    attn = torch.tensor(_pad_to_len(test["attention_mask"]))
    y = torch.tensor(test["label_id"])
    seg = (torch.tensor(_pad_to_len(test["segment_ids"]))
           if use_dual else None)

    loader = DataLoader(
        TensorDataset(input_ids, attn, y),
        batch_size=64,
        shuffle=False,
    )
    seen = 0
    with torch.no_grad():
        for ids, am, yb in loader:
            kwargs = {"input_ids": ids.to(device), "attention_mask": am.to(device)}
            if use_dual:
                kwargs["segment_ids"] = seg[seen: seen + ids.size(0)].to(device)
            out = model(**kwargs)
            logits = out.logits if hasattr(out, "logits") else out
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(yb.tolist())
            seen += ids.size(0)

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "tag": args.tag,
        "accuracy": accuracy(labels, preds),
        "macro_f1": macro_f1(labels, preds),
        "n_test": len(labels),
    }
    print(json.dumps(summary, indent=2))

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if args.report:
        report = classification_report_dict(labels, preds, PRIMARY_LABELS)
        cm = confusion_matrix_df(labels, preds, PRIMARY_LABELS)
        with open(out_dir / "classification_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        cm.to_csv(out_dir / "confusion_matrix.csv")
        print("\nPer-class report:")
        for name in PRIMARY_LABELS:
            r = report[name]
            print(f"  {name:<12} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f}")
        print("\nConfusion matrix:\n", cm)


if __name__ == "__main__":
    main()