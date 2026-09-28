#!/usr/bin/env python3
"""Per-instance error analysis for the citation-intent classifiers.

Dumps one row per test instance (gold, prediction, context length, section,
retrieved evidence and its labels) and aggregates the error structure along the
axes that matter for a scientometric readership:

  * which intent pairs are confused (confusion matrix, top confusions),
  * per-class precision / recall / F1,
  * accuracy by section of the citing paper,
  * accuracy by citation-context length,
  * accuracy conditioned on whether the retrieved neighbours carry the gold
    label (i.e. whether a label-copying path was even available).

Usage:
    python scripts/error_analysis.py --checkpoint runs/verify_fix/best_model \
        --tag err_baseline_deberta
    python scripts/error_analysis.py --checkpoint runs/real_deberta_dual_k3_01/best_model \
        --index data/index/scicite_bm25l_train.pkl --tag err_dual_bm25l
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from src.data.dataset import PRIMARY_LABELS, PRIMARY_ID2LABEL, read_jsonl
from src.models.classifier import build_model
from src.retrieval.retrieval_augmented import RetrievalAugmenter
from src.training.utils import get_tokenizer, tokenize_augmented

MODEL_FOR_CHECKPOINT = "deberta"

SECTION_RULES = [
    ("Introduction", ("introduction", "intro")),
    ("Related work", ("related work", "relatedwork", "background", "literature")),
    ("Methods", ("method", "materials and methods", "methodology", "experimental setup",
                 "implementation", "study design", "experimental design", "approach")),
    ("Results", ("result", "experiment", "evaluation", "accuracy", "performance")),
    ("Discussion", ("discussion", "conclusion", "limitation", "future work")),
]


def section_bucket(raw) -> str:
    s = "" if raw is None else str(raw)
    if s.lower() in ("nan", "none"):
        s = ""
    s = s.strip().lower()
    s = re.sub(r"^[0-9]+(\.[0-9]+)*[.)]?\s*", "", s)
    for bucket, keys in SECTION_RULES:
        for k in keys:
            if k in s:
                return bucket
    return "Other / unlabelled"


def length_bucket(n_words: int) -> str:
    if n_words <= 20:
        return "<=20 words"
    if n_words <= 40:
        return "21-40 words"
    if n_words <= 60:
        return "41-60 words"
    return ">60 words"


def pad_to_len(seqs):
    maxlen = max(len(s) for s in seqs)
    return [list(s) + [0] * (maxlen - len(s)) for s in seqs]


def _load_weights(checkpoint: Path) -> dict:
    st = checkpoint / "model.safetensors"
    if st.exists():
        from safetensors.torch import load_file

        return load_file(str(st))
    binp = checkpoint / "pytorch_model.bin"
    if binp.exists():
        return torch.load(binp, map_location="cpu", weights_only=True)
    raise FileNotFoundError(f"no model weights found in {checkpoint}")


def load_model(checkpoint: Path, pooling: str, num_labels: int):
    cap = checkpoint / "head_config.json"
    if cap.exists():
        pooling = json.loads(cap.read_text(encoding="utf-8")).get("pooling", pooling)
    if pooling == "dual":
        model = build_model(MODEL_FOR_CHECKPOINT, num_labels=num_labels, pooling="dual")
        state = torch.load(checkpoint / "dual_head.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model, "dual"

    # The plain [CLS] runs are saved by ``Trainer.save_model`` on a custom
    # ``nn.Module``, which writes weights but no ``config.json``, so
    # ``AutoModelForSequenceClassification.from_pretrained`` cannot read them.
    # Rebuild the module from the backbone and load the raw state dict.
    model = build_model(MODEL_FOR_CHECKPOINT, num_labels=num_labels, pooling="cls")
    model.load_state_dict(_load_weights(checkpoint))
    return model, "cls"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default="data/processed/scicite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", default="data/index/scicite_bm25l_train.pkl")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    rows = read_jsonl(Path(args.data) / f"{args.split}.jsonl")
    tokenizer = get_tokenizer(MODEL_FOR_CHECKPOINT)

    model, pooling = load_model(checkpoint, "cls", num_labels=len(PRIMARY_LABELS))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # ---- build inputs (and, for the dual head, the evidence + its labels) ----
    index = None
    if pooling == "dual":
        with open(args.index, "rb") as fh:
            index = pickle.load(fh)
        augmenter = RetrievalAugmenter(
            index=index, top_k=args.top_k, leak_free=False, labels=PRIMARY_LABELS
        )
        text2idx = getattr(index, "_text2idx", {})
        idx_labels = getattr(index, "labels", None)

    records, feats = [], []
    for i, r in enumerate(rows):
        gold = r["label"]
        gold_id = PRIMARY_LABELS.index(gold)
        rec = {
            "row": i,
            "id": r.get("id") or f"{args.split}-{i}",
            "gold": gold,
            "n_words": len(str(r["string"]).split()),
            "section_raw": r.get("sectionName", ""),
            "section": section_bucket(r.get("sectionName", "")),
        }
        if pooling == "dual":
            aug = augmenter.augment(r["string"], target_label=gold, doc_id=r.get("id", ""))
            evidence = aug["evidence"]
            ev_labels = []
            for ev in evidence:
                if idx_labels is not None and ev in text2idx:
                    ev_labels.append(str(idx_labels[text2idx[ev][0]]))
                else:
                    ev_labels.append("unknown")
            f = tokenize_augmented(tokenizer, aug["target"], evidence, args.max_length)
            rec["evidence_labels"] = ev_labels
            rec["gold_in_evidence"] = gold in ev_labels
            rec["n_evidence"] = len(evidence)
            rec["evidence"] = evidence
        else:
            enc = tokenizer(r["string"], truncation=True, max_length=args.max_length)
            f = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
            rec["n_tokens"] = len(enc["input_ids"])
        rec["label_id"] = gold_id
        records.append(rec)
        feats.append(f)

    input_ids = torch.tensor(pad_to_len([f["input_ids"] for f in feats]))
    attn = torch.tensor(pad_to_len([f["attention_mask"] for f in feats]))
    seg = (torch.tensor(pad_to_len([f["segment_ids"] for f in feats]))
           if pooling == "dual" else None)

    preds = []
    with torch.no_grad():
        for start in range(0, len(records), 64):
            end = start + 64
            kwargs = {
                "input_ids": input_ids[start:end].to(device),
                "attention_mask": attn[start:end].to(device),
            }
            if pooling == "dual":
                kwargs["segment_ids"] = seg[start:end].to(device)
            out = model(**kwargs)
            logits = out.logits if hasattr(out, "logits") else out
            preds.extend(logits.argmax(-1).cpu().tolist())

    for rec, p in zip(records, preds):
        rec["pred"] = PRIMARY_ID2LABEL[int(p)]
        rec["correct"] = rec["pred"] == rec["gold"]

    # ------------------------------------------------------------------ stats
    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    conf = Counter((r["gold"], r["pred"]) for r in records if not r["correct"])
    cm = defaultdict(Counter)
    for r in records:
        cm[r["gold"]][r["pred"]] += 1

    per_class = {}
    for lbl in PRIMARY_LABELS:
        tp = sum(1 for r in records if r["gold"] == lbl and r["pred"] == lbl)
        fp = sum(1 for r in records if r["gold"] != lbl and r["pred"] == lbl)
        fn = sum(1 for r in records if r["gold"] == lbl and r["pred"] != lbl)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec_ = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
        per_class[lbl] = {
            "support": sum(1 for r in records if r["gold"] == lbl),
            "precision": round(prec, 4),
            "recall": round(rec_, 4),
            "f1": round(f1, 4),
        }

    def grouped(key_fn):
        buckets = defaultdict(lambda: [0, 0])
        for r in records:
            k = key_fn(r)
            buckets[k][0] += 1
            buckets[k][1] += int(r["correct"])
        return {
            k: {"n": v[0], "correct": v[1], "accuracy": round(v[1] / v[0], 4)}
            for k, v in sorted(buckets.items())
        }

    summary = {
        "tag": args.tag,
        "checkpoint": str(checkpoint),
        "pooling": pooling,
        "n_test": n,
        "accuracy": round(acc, 4),
        "confusion_matrix": {g: dict(c) for g, c in cm.items()},
        "per_class": per_class,
        "top_confusions": [
            {"gold": g, "pred": p, "n": c} for (g, p), c in conf.most_common(6)
        ],
        "by_section": grouped(lambda r: r["section"]),
        "by_length": grouped(lambda r: length_bucket(r["n_words"])),
    }
    if pooling == "dual":
        summary["by_gold_in_evidence"] = grouped(lambda r: str(r["gold_in_evidence"]))
        summary["mean_evidence_agreement"] = round(
            sum(1 for r in records if r["gold_in_evidence"]) / n, 4
        )

    out_dir = Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "error_analysis.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    cols = ["row", "id", "gold", "pred", "correct", "n_words", "section",
            "section_raw", "n_evidence", "gold_in_evidence", "evidence_labels"]
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in cols}
            if isinstance(row.get("evidence_labels"), list):
                row["evidence_labels"] = "|".join(row["evidence_labels"])
            w.writerow(row)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()