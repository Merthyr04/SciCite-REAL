#!/usr/bin/env python3
"""Preprocess the ACL-ARC release shipped with SciCite into pipeline format.

ACL-ARC (Jurgens et al., TACL 2018) is distributed alongside SciCite by the
Allen Institute for AI. Its raw records use different field names from SciCite
and carry a six-way ``intent`` label:

    Background | Uses | CompareOrContrast | Motivation | Extends | Future

We normalise it to the same schema the rest of the pipeline consumes
(``string`` / ``label`` / ``sectionName`` / ``id``), keeping the six-way label
intact rather than collapsing it onto SciCite's three classes, so that the
cross-domain experiment needs no mapping assumptions.

Usage:
    python scripts/preprocess_aclarc.py \
        --raw data/raw/acl-arc/acl-arc --out data/processed/aclarc
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.dataset import read_jsonl


def _clean(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none") else s


def preprocess(raw_dir: str | Path, out_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    sections = Counter()
    total = 0

    for split in ("train", "dev", "test"):
        rows = read_jsonl(raw_dir / f"{split}.jsonl")
        clean = []
        for i, r in enumerate(rows):
            string = _clean(r.get("text") or r.get("cleaned_cite_text"))
            label = _clean(r.get("intent"))
            if not string or not label:
                continue
            rec = {
                "id": _clean(r.get("citation_id"))
                or f"{_clean(r.get('citing_paper_id'))}#{r.get('citation_excerpt_index', i)}",
                "string": string,
                "sectionName": _clean(r.get("section_name")),
                "label": label,
                "citingPaperId": _clean(r.get("citing_paper_id")),
                "citedPaperId": _clean(r.get("cited_paper_id")),
                "source": "acl-arc",
            }
            clean.append(rec)
            counts[label] += 1
            sections[rec["sectionName"]] += 1
            total += 1

        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for rec in clean:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{split}] {len(clean)} rows")

    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "total": total,
                "intent_counts": dict(counts.most_common()),
                "section_counts": dict(sections.most_common()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nIntent balance: {dict(counts.most_common())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/acl-arc/acl-arc")
    ap.add_argument("--out", default="data/processed/aclarc")
    args = ap.parse_args()
    preprocess(args.raw, args.out)


if __name__ == "__main__":
    main()