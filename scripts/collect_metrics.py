#!/usr/bin/env python3
"""Collate all run metrics into one markdown table for the paper.

Usage:
    python scripts/collect_metrics.py runs
    python scripts/collect_metrics.py runs --out results_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(root: str) -> list:
    rows = []
    for p in sorted(Path(root).glob("*/metrics.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        row = {
            "tag": data.get("tag", p.parent.name),
            "model": data.get("model", ""),
            "retrieval": data.get("retrieval", False),
            "pooling": data.get("pooling", "cls"),
            "top_k": data.get("top_k", ""),
            "leak_free": data.get("leak_free", ""),
            "test_accuracy": round(data.get("test_accuracy", 0), 4),
            "test_macro_f1": round(data.get("test_macro_f1", 0), 4),
            "val_accuracy": round(data.get("eval_accuracy", 0), 4),
            "val_macro_f1": round(data.get("eval_macro_f1", 0), 4),
        }
        rows.append(row)
    return rows


def to_markdown(rows: list) -> str:
    header = "| tag | model | retrieval | pooling | top_k | leak_free | TestAcc | Macro-F1 |\n"
    sep = "|-----|-------|-----------|---------|-------|-----------|---------|----------|\n"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['model']} | {r['retrieval']} | {r['pooling']} "
            f"| {r['top_k']} | {r['leak_free']} | {r['test_accuracy']} | {r['test_macro_f1']} |\n"
        )
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", default="runs")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = collect(args.root)
    table = to_markdown(rows)
    print(table)
    if args.out:
        Path(args.out).write_text(table, encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()