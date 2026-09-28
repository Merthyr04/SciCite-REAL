"""Dump the exact numbers the manuscript claims, for side-by-side checking."""
from __future__ import annotations

import json
from pathlib import Path


def show(title: str, path: str, keys: list[str]) -> None:
    print(f"\n########## {title} ({path})")
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for k in keys:
        print(f"--- {k} ---")
        print(json.dumps(d.get(k), indent=1, ensure_ascii=False)[:2200])


show(
    "SciCite distribution",
    "runs/distribution_analysis.json",
    ["label_distribution", "priors_on_test", "context_length_words"],
)
show(
    "SciCite section table",
    "runs/distribution_analysis.json",
    ["section_distribution", "intent_by_section_train"],
)
show(
    "ACL-ARC",
    "runs/distribution_aclarc.json",
    ["priors_on_test", "section_distribution", "intent_by_section_train"],
)