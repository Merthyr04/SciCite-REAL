#!/usr/bin/env python3
"""Descriptive analysis of citation-intent structure in SciCite.

Produces the material for the "citation intent distribution" section: how the
three intents are distributed overall and across splits, how they distribute
over the rhetorical sections of citing papers, how long the contexts are per
intent, and how much of the intent signal a *section prior* alone explains.

The section prior is a deliberately dumb baseline (predict the majority intent
observed in that section on the training split). It quantifies how much of the
task is explained by document structure rather than by sentence content, which
is the reference point any automatic classifier has to clear.

Usage:
    python scripts/distribution_analysis.py --data data/processed/scicite \
        --out runs/distribution_analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from src.data.dataset import PRIMARY_LABELS, read_jsonl

SECTION_RULES = [
    ("Introduction", ("introduction", "intro")),
    ("Related work", ("related work", "relatedwork", "background", "literature")),
    ("Methods", ("method", "materials and methods", "methodology", "experimental setup",
                 "implementation", "study design", "experimental design", "approach")),
    ("Results", ("result", "experiment", "evaluation", "accuracy", "performance")),
    ("Discussion", ("discussion", "conclusion", "limitation", "future work")),
]
SECTION_ORDER = [b for b, _ in SECTION_RULES] + ["Other / unlabelled"]


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


def load_split(data_dir: Path, name: str):
    rows = read_jsonl(data_dir / f"{name}.jsonl")
    for r in rows:
        r["_section"] = section_bucket(r.get("sectionName", ""))
        r["_words"] = len(str(r["string"]).split())
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/scicite")
    ap.add_argument("--out", default="runs/distribution_analysis.json")
    args = ap.parse_args()

    data_dir = Path(args.data)
    splits = {n: load_split(data_dir, n) for n in ("train", "dev", "test")}
    train, test = splits["train"], splits["test"]

    # Label set is inferred from the data so the same script serves the
    # three-class SciCite task and the six-class ACL-ARC task.
    labels = sorted({r["label"] for r in train})

    out = {}

    # ---- 1. overall label distribution ----
    out["label_distribution"] = {}
    for name, rows in splits.items():
        c = Counter(r["label"] for r in rows)
        n = len(rows)
        out["label_distribution"][name] = {
            lbl: {"n": c.get(lbl, 0), "pct": round(100 * c.get(lbl, 0) / n, 2)}
            for lbl in labels
        }
        out["label_distribution"][name]["_total"] = n

    # ---- 2. section distribution ----
    out["section_distribution"] = {}
    for name, rows in splits.items():
        c = Counter(r["_section"] for r in rows)
        n = len(rows)
        out["section_distribution"][name] = {
            s: {"n": c.get(s, 0), "pct": round(100 * c.get(s, 0) / n, 2)}
            for s in SECTION_ORDER if c.get(s, 0) > 0
        }

    # ---- 3. intent x section cross-tab (row-normalised over sections) ----
    cross = {s: Counter() for s in SECTION_ORDER}
    for r in train:
        cross[r["_section"]][r["label"]] += 1
    out["intent_by_section_train"] = {}
    for s in SECTION_ORDER:
        tot = sum(cross[s].values())
        if not tot:
            continue
        out["intent_by_section_train"][s] = {
            "n": tot,
            "pct_of_split": round(100 * tot / len(train), 2),
            **{lbl: round(100 * cross[s].get(lbl, 0) / tot, 1) for lbl in labels},
        }

    # ---- 4. context length per intent ----
    out["context_length_words"] = {}
    for lbl in labels:
        ws = [r["_words"] for r in train if r["label"] == lbl]
        out["context_length_words"][lbl] = {
            "n": len(ws),
            "mean": round(statistics.mean(ws), 1),
            "median": statistics.median(ws),
            "p90": sorted(ws)[int(0.9 * len(ws))],
            "max": max(ws),
        }

    # ---- 5. priors: global majority vs section prior (fit on train, test on test) ----
    global_majority = Counter(r["label"] for r in train).most_common(1)[0][0]
    section_majority = {}
    for s in SECTION_ORDER:
        c = cross[s]
        if c:
            section_majority[s] = c.most_common(1)[0][0]

    global_correct = sum(1 for r in test if r["label"] == global_majority)
    prior_correct = 0
    fallback_used = 0
    for r in test:
        pred = section_majority.get(r["_section"])
        if pred is None:
            pred = global_majority
            fallback_used += 1
        prior_correct += int(pred == r["label"])

    out["priors_on_test"] = {
        "global_majority_label": global_majority,
        "global_majority_accuracy": round(global_correct / len(test), 4),
        "section_prior_accuracy": round(prior_correct / len(test), 4),
        "section_prior_fallback_rows": fallback_used,
        "section_majority_map": section_majority,
        "n_test": len(test),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()