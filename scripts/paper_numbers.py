#!/usr/bin/env python3
"""Collate every number the manuscript needs into one JSON blob.

Reads runs/*/metrics.json and runs/*/error_analysis.json and emits the exact
values that go into each table of the Scientometrics manuscript, so the tables
can be filled without transcription errors.

Usage:
    python scripts/paper_numbers.py --out runs/paper_numbers.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = "runs"

# tag -> label used in the manuscript tables
MAIN = {
    "baseline_bert": "BERT-base",
    "baseline_scibert": "SciBERT",
    "baseline_deberta": "DeBERTa-v3-base",
    "baseline_deberta_s123": "DeBERTa-v3-base (seed 123)",
    "verify_fix": "DeBERTa-v3-base (repeat run)",
    "real_deberta_k3": "DeBERTa + BM25L (CLS)",
    "real_deberta_dual_k3_01": "DeBERTa + BM25L (dual, run 2)",
    "real_deberta_dual_k3": "DeBERTa + BM25L (dual, run 1)",
    "real_deberta_dual_k3_bm25l_d02": "DeBERTa + BM25L (dual, dropout 0.2)",
    "real_deberta_dual_k3_bm25l_s1": "DeBERTa + BM25L (dual, seed 1)",
    "real_deberta_dual_k3_bm25": "DeBERTa + BM25 (dual, seed 42)",
    "real_deberta_dual_k3_bm25_s1": "DeBERTa + BM25 (dual, seed 1)",
    "real_deberta_k3_bm25": "DeBERTa + BM25 (CLS)",
    "real_deberta_k3_pt": "DeBERTa + BM25L (CLS, target preserved)",
    "real_deberta_k3_bm25_pt": "DeBERTa + BM25 (CLS, target preserved)",
    "bert_ret_dual": "BERT + BM25L (dual)",
    "scibert_ret_dual": "SciBERT + BM25L (dual)",
}

# ``real_deberta_k3`` and ``real_deberta_k3_bm25`` join evidence in front of the
# target and truncate from the right, which deletes the target sentence in 95.9%
# and 5.4% of training rows respectively; the ``_pt`` runs keep the same [CLS]
# head and change only that truncation policy. ``diag_fp32_nockpt`` and
# ``smoke2`` are under-trained diagnostics (1 epoch) and no table cites them.
ABLATION = {
    "real_deberta_k3": "CLS head (right-truncated)",
    "real_deberta_k3_pt": "CLS head (target preserved)",
    "real_deberta_dual_k3_01": "Dual head",
    "ablation_real_deberta_leakfree": "Leak-free mask",
}

ACLARC = {
    "aclarc_deberta": "DeBERTa-v3-base (no retrieval)",
    "aclarc_ret_cls": "+ BM25L (CLS head)",
    "aclarc_ret_dual": "+ BM25L (dual head)",
    "aclarc_ret_leakfree": "+ BM25L (dual, leak-free)",
}

ERRORS = {
    "err_baseline_deberta": "DeBERTa-v3-base baseline",
    "err_dual_bm25l": "DeBERTa + BM25L (dual)",
    "err_dual_bm25l_d02": "DeBERTa + BM25L (dual, d=0.2)",
    "err_dual_bm25": "DeBERTa + BM25 (dual)",
}


def metrics(tag: str) -> dict | None:
    p = Path(RUNS) / tag / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "acc": round(100 * d.get("test_accuracy", 0), 2),
        "f1": round(100 * d.get("test_macro_f1", 0), 2),
        "val_acc": round(100 * d.get("eval_accuracy", 0), 2),
        "val_f1": round(100 * d.get("eval_macro_f1", 0), 2),
        "epochs": d.get("epoch"),
        "pooling": d.get("pooling"),
        "retrieval": d.get("retrieval"),
        "seed": d.get("seed"),
        "leak_free": d.get("leak_free"),
        "train_acc": round(100 * d.get("train_accuracy", 0), 2),
    }


def errors(tag: str) -> dict | None:
    p = Path(RUNS) / tag / "error_analysis.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "n_test": d["n_test"],
        "accuracy": round(100 * d["accuracy"], 2),
        "per_class": d["per_class"],
        "by_section": d["by_section"],
        "by_length": d["by_length"],
        "top_confusions": d["top_confusions"],
        "confusion_matrix": d["confusion_matrix"],
        "by_gold_in_evidence": d.get("by_gold_in_evidence"),
        "mean_evidence_agreement": d.get("mean_evidence_agreement"),
    }


def collect(group: dict) -> dict:
    out = {}
    for tag, label in group.items():
        m = metrics(tag)
        if m is None:
            out[label] = {"__missing__": tag}
        else:
            out[label] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper_numbers.json")
    args = ap.parse_args()

    blob = {
        "main": collect(MAIN),
        "ablation": collect(ABLATION),
        "aclarc": collect(ACLARC),
        "error_analysis": {lbl: errors(tag) for tag, lbl in ERRORS.items()},
    }

    # ACL-ARC / SciCite distribution summaries, if present.
    for name, path in (
        ("distribution_scicite", "runs/distribution_analysis.json"),
        ("distribution_aclarc", "runs/distribution_aclarc.json"),
    ):
        p = Path(path)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            blob[name] = {
                "label_distribution": d["label_distribution"],
                "intent_by_section_train": d["intent_by_section_train"],
                "priors_on_test": d["priors_on_test"],
                "section_distribution": d["section_distribution"],
                "context_length_words": d["context_length_words"],
            }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")

    missing = [
        f"{grp}:{lbl}({v['__missing__']})"
        for grp in ("main", "ablation", "aclarc")
        for lbl, v in blob[grp].items()
        if "__missing__" in v
    ]
    missing += [f"errors:{k}" for k, v in blob["error_analysis"].items() if v is None]
    print(f"[saved] {args.out}")
    print(f"[missing] {missing if missing else 'none'}")


if __name__ == "__main__":
    main()