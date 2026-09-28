#!/usr/bin/env python3
"""Inject the collated run numbers into the Scientometrics manuscript.

Reads ``runs/paper_numbers.json`` (produced by ``scripts/paper_numbers.py``) and
replaces every ALL_CAPS placeholder in ``paper/scientometrics/sn_scicite.tex``
with the measured value, so the tables cannot drift from the logs.

Usage:
    python scripts/paper_numbers.py --out runs/paper_numbers.json
    python scripts/fill_paper.py                 # writes .tex in place
    python scripts/fill_paper.py --check         # report leftovers, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TEX = Path("paper/scientometrics/sn_scicite.tex")
NUMBERS = Path("runs/paper_numbers.json")

# Prose is written by hand, not injected from the logs.
PROSE = {"ACLARC_PROSE"}


def pct(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=str(TEX))
    ap.add_argument("--numbers", default=str(NUMBERS))
    ap.add_argument("--check", action="store_true",
                    help="only report which placeholders would remain")
    args = ap.parse_args()

    blob = json.loads(Path(args.numbers).read_text(encoding="utf-8"))
    main_t, abl, aclarc = blob["main"], blob["ablation"], blob["aclarc"]
    err = blob["error_analysis"]

    def get(group: dict, label: str, field: str):
        entry = group.get(label)
        if entry is None or "__missing__" in entry:
            return None
        return entry.get(field)

    vals: dict[str, str] = {}

    def put(name: str, value):
        if value is not None:
            vals[name] = str(value)

    # ---- main comparison table -------------------------------------------
    for ph, label, tag in (
        ("MAIN_DUAL1", "DeBERTa + BM25L (dual, run 1)", "real_deberta_dual_k3"),
        ("MAIN_DUAL2", "DeBERTa + BM25L (dual, run 2)", "real_deberta_dual_k3_01"),
    ):
        a = get(main_t, label, "acc")
        f = get(main_t, label, "f1")
        if a is None:
            print(f"[warn] main run not available yet: {tag}")
        put(f"{ph}_ACC", pct(a) if a is not None else None)
        put(f"{ph}_F1", pct(f) if f is not None else None)

    # ---- augmented runs on the other two backbones -----------------------
    for ph, label in (
        ("BERT_DUAL", "BERT + BM25L (dual)"),
        ("SCIBERT_DUAL", "SciBERT + BM25L (dual)"),
    ):
        a = get(main_t, label, "acc")
        f = get(main_t, label, "f1")
        if a is None:
            print(f"[warn] backbone run not available yet: {label}")
        put(f"{ph}_ACC", pct(a) if a is not None else None)
        put(f"{ph}_F1", pct(f) if f is not None else None)

    # ---- retrieval-variant table -----------------------------------------
    for ph, label in (
        ("BM25_CLS", "DeBERTa + BM25 (CLS)"),
        ("BM25_DUAL", "DeBERTa + BM25 (dual, seed 42)"),
        ("BM25_DUAL_S1", "DeBERTa + BM25 (dual, seed 1)"),
        ("BM25L_DUAL_S1", "DeBERTa + BM25L (dual, seed 1)"),
        ("BM25L_CLS_PT", "DeBERTa + BM25L (CLS, target preserved)"),
        ("BM25_CLS_PT", "DeBERTa + BM25 (CLS, target preserved)"),
    ):
        a = get(main_t, label, "acc")
        f = get(main_t, label, "f1")
        if a is None:
            print(f"[warn] variant run not available yet: {label}")
        put(f"{ph}_ACC", pct(a) if a is not None else None)
        put(f"{ph}_F1", pct(f) if f is not None else None)

    # ---- head / truncation / leakage ablation ----------------------------
    pt_acc = get(abl, "CLS head (target preserved)", "acc")
    for ph, label in (
        ("BM25L_CLS_PT", "CLS head (target preserved)"),
        ("LEAK", "Leak-free mask"),
    ):
        a = get(abl, label, "acc")
        f = get(abl, label, "f1")
        if a is None:
            print(f"[warn] ablation run not available yet: {label}")
        put(f"{ph}_ACC", pct(a) if a is not None else None)
        put(f"{ph}_F1", pct(f) if f is not None else None)

    # Gap closed by preserving the target, against the right-truncated CLS row.
    rt_acc = get(abl, "CLS head (right-truncated)", "acc")
    if pt_acc is not None and rt_acc is not None:
        put("TOK_HEAD_GAP", pct(pt_acc - rt_acc, 1))
    else:
        print("[warn] cannot compute TOK_HEAD_GAP yet")

    # ---- per-class + by-section (DeBERTa baseline error analysis) --------
    base = err.get("DeBERTa-v3-base baseline")
    if base is None:
        print("[warn] error analysis for the DeBERTa baseline is missing")
    else:
        pc = base["per_class"]
        for ph, lbl in (("BG", "background"), ("ME", "method"), ("RE", "result")):
            c = pc.get(lbl, {})
            put(f"PERCLASS_{ph}_SUP", c.get("support"))
            put(f"PERCLASS_{ph}_P", pct(100 * c["precision"]) if "precision" in c else None)
            put(f"PERCLASS_{ph}_R", pct(100 * c["recall"]) if "recall" in c else None)
            put(f"PERCLASS_{ph}_F1", pct(100 * c["f1"]) if "f1" in c else None)
        for ph, sec in (
            ("INTRO", "Introduction"),
            ("REL", "Related work"),
            ("METH", "Methods"),
            ("RES", "Results"),
            ("DISC", "Discussion"),
            ("OTH", "Other / unlabelled"),
        ):
            s = base["by_section"].get(sec)
            put(f"ERRSEC_{ph}_N", s["n"] if s else None)
            put(f"ERRSEC_{ph}_A", pct(100 * s["accuracy"]) if s else None)
        for ph, bucket in (
            ("S", "<=20 words"),
            ("M", "21-40 words"),
            ("L", "41-60 words"),
            ("XL", ">60 words"),
        ):
            b = base["by_length"].get(bucket)
            put(f"ERRLEN_{ph}_N", b["n"] if b else None)
            put(f"ERRLEN_{ph}_A", pct(100 * b["accuracy"]) if b else None)

    # ---- ACL-ARC cross-domain table --------------------------------------
    for ph, label in (
        ("ACLARC_BASE", "DeBERTa-v3-base (no retrieval)"),
        ("ACLARC_CLS", "+ BM25L (CLS head)"),
        ("ACLARC_DUAL", "+ BM25L (dual head)"),
        ("ACLARC_LEAK", "+ BM25L (dual, leak-free)"),
    ):
        a = get(aclarc, label, "acc")
        f = get(aclarc, label, "f1")
        if a is None:
            print(f"[warn] ACL-ARC run not available yet: {label}")
        put(f"{ph}_ACC", pct(a) if a is not None else None)
        put(f"{ph}_F1", pct(f) if f is not None else None)

    # ---- substitute -------------------------------------------------------
    tex_path = Path(args.tex)
    tex = tex_path.read_text(encoding="utf-8")

    def repl(m: re.Match) -> str:
        token = m.group(0)
        return vals.get(token, token)

    token_re = re.compile(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b")
    out = token_re.sub(repl, tex)

    leftover = sorted(
        {t for t in token_re.findall(out) if t not in PROSE}
    )
    unfilled = [t for t in leftover if t not in vals]

    if args.check:
        print(f"[check] placeholders still unresolved: {unfilled or 'none'}")
        print(f"[check] hand-written prose markers: {sorted(PROSE)}")
        return

    tex_path.write_text(out, encoding="utf-8")
    print(f"[saved] {tex_path}")
    print(f"[filled] {len(vals)} values")
    if unfilled:
        print(f"[MISSING] {unfilled}")
    else:
        print("[MISSING] none")


if __name__ == "__main__":
    main()