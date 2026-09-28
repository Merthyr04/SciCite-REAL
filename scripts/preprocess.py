#!/usr/bin/env python3
"""Preprocess raw SciCite jsonl into clean train/dev/test files.

The raw tar.gz ships with ``train.jsonl`` / ``dev.jsonl`` / ``test.jsonl`` plus
per-split ``*_metadata.json`` files. We keep the three primary splits and write
a compact ``*.clean.jsonl`` + ``stats.json`` for reproducibility, and also emit
an optional fine-grained ``*_label2.jsonl`` for the 4-class experiment.

Usage:
    python scripts/preprocess.py --raw data/raw/scicite --out data/processed/scicite
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.dataset import read_jsonl


def preprocess(raw_dir: str | Path, out_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": "train", "dev": "dev", "test": "test"}

    class_counter = Counter()
    fine_counter = Counter()
    total = 0
    for split, fname in splits.items():
        rows = read_jsonl(raw_dir / f"{fname}.jsonl")
        clean = []
        fine = []
        for r in rows:
            string = (r.get("context") or r.get("string") or "").strip()
            if not string:
                continue
            label = str(r.get("label", "")).strip()
            label2 = str(r.get("label2", "") if r.get("label2") is not None else "").strip()
            rec = {
                "id": str(r.get("id", r.get("excerpt_index", ""))),
                "string": string,
                "sectionName": r.get("sectionName", ""),
                "label": label,
                "citingPaperId": r.get("citingPaperId"),
                "citedPaperId": r.get("citedPaperId"),
                "source": r.get("source"),
            }
            clean.append(rec)
            class_counter[label] += 1
            total += 1
            if label2:
                fine.append({**rec, "label2": label2})
                fine_counter[label2] += 1

        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for rec in clean:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if fine:
            with open(out_dir / f"{split}_label2.jsonl", "w", encoding="utf-8") as fh:
                for rec in fine:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{split}] {len(clean)} rows")

    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "total": total,
                "label_counts": dict(class_counter.most_common()),
                "label2_counts": dict(fine_counter.most_common()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nClass balance: {dict(class_counter.most_common())}")
    print(f"Fine-grained (label2): {dict(fine_counter.most_common())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/scicite")
    ap.add_argument("--out", default="data/processed/scicite")
    args = ap.parse_args()
    preprocess(args.raw, args.out)


if __name__ == "__main__":
    main()