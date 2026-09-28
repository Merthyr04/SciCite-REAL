"""Benchmark: rank_bm25.BM25L vs FastBM25L on the SciCite train split.

Measures (1) score/ranking equivalence of the vectorised scorer against the
reference implementation and (2) the wall-clock cost of the full retrieval
preprocessing pass each backend needs for all train queries.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from rank_bm25 import BM25L

from src.data.preprocessing import tokenize_for_bm25
from src.retrieval.fast_bm25l import FastBM25L


def load_train():
    rows = []
    with open(ROOT / "data/processed/scicite/train.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def top_k_texts(model, q_tokens, corpus, k=3):
    scores = model.get_scores(q_tokens)
    order = np.argsort(-scores)[:k]
    return [corpus[i] for i in order]


def main():
    rows = load_train()
    corpus = [r["string"] for r in rows]
    print(f"[bench] {len(corpus)} train contexts")

    t0 = time.perf_counter()
    tok_docs = [tokenize_for_bm25(d) for d in corpus]
    print(f"[bench] tokenisation: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    ref = BM25L(tok_docs, k1=1.5, b=0.75, delta=0.5)
    t_ref_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = FastBM25L(tok_docs, k1=1.5, b=0.75, delta=0.5)
    t_fast_build = time.perf_counter() - t0
    print(f"[bench] index build: rank_bm25 {t_ref_build:.2f}s | fast {t_fast_build:.2f}s")

    tok_queries = [tokenize_for_bm25(q) for q in corpus]

    # -- equivalence on all queries (reference timing included) -------------
    max_diff = 0.0
    top3_mismatch = 0
    t0 = time.perf_counter()
    ref_scores_all = []
    for q in tok_queries:
        ref_scores_all.append(ref.get_scores(q))
    t_ref_scores = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast_scores_all = [fast.get_scores(q) for q in tok_queries]
    t_fast_scores = time.perf_counter() - t0

    for rs, fs in zip(ref_scores_all, fast_scores_all):
        max_diff = max(max_diff, float(np.max(np.abs(rs - fs))))
        if list(np.argsort(-rs)[:3]) != list(np.argsort(-fs)[:3]):
            top3_mismatch += 1
    print(
        f"[bench] equivalence: max|Δscore|={max_diff:.3e} | "
        f"top-3 mismatches: {top3_mismatch}/{len(tok_queries)}"
    )

    # -- end-to-end augmentation pass via BM25Index -------------------------
    from src.retrieval.index import BM25Index
    from src.retrieval.retrieval_augmented import RetrievalAugmenter, batch_augment

    t0 = time.perf_counter()
    idx_fast = BM25Index(corpus, variant="bm25l")
    t_idx_fast_build = time.perf_counter() - t0
    aug = RetrievalAugmenter(index=idx_fast, top_k=3, evidence_source="train")
    instances = [{"string": r["string"], "label": r["label"]} for r in rows]
    t0 = time.perf_counter()
    augmented_fast = batch_augment(aug, instances)
    t_aug_fast = time.perf_counter() - t0
    print(f"[bench] fast augment pass: {t_aug_fast:.2f}s (index build {t_idx_fast_build:.2f}s)")

    # reference augment pass with rank_bm25 backend, monkey-patched in
    idx_ref = BM25Index(corpus, variant="bm25l")
    idx_ref._model = ref  # swap in the reference scorer
    aug_ref = RetrievalAugmenter(index=idx_ref, top_k=3, evidence_source="train")
    t0 = time.perf_counter()
    augmented_ref = batch_augment(aug_ref, instances)
    t_aug_ref = time.perf_counter() - t0
    print(f"[bench] rank_bm25 augment pass: {t_aug_ref:.2f}s")

    same = sum(
        1
        for a, b in zip(augmented_fast, augmented_ref)
        if a["evidence"] == b["evidence"]
    )
    print(f"[bench] evidence identical: {same}/{len(instances)}")

    out = {
        "n_queries": len(corpus),
        "rank_bm25": {
            "index_build_s": round(t_ref_build, 3),
            "scores_pass_s": round(t_ref_scores, 3),
            "augment_pass_s": round(t_aug_ref, 3),
        },
        "fast_bm25l": {
            "index_build_s": round(t_fast_build, 3),
            "scores_pass_s": round(t_fast_scores, 3),
            "augment_pass_s": round(t_aug_fast, 3),
            "bm25index_build_s": round(t_idx_fast_build, 3),
        },
        "max_abs_score_diff": max_diff,
        "top3_mismatches": top3_mismatch,
        "evidence_identical": same,
        "speedup_scores_pass": round(t_ref_scores / t_fast_scores, 1),
        "speedup_augment_pass": round(t_aug_ref / t_aug_fast, 1),
    }
    out_path = ROOT / "runs" / "fast_bm25l_bench.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[bench] wrote {out_path}")


if __name__ == "__main__":
    main()
