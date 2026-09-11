"""Vectorised BM25L scoring with identical semantics to ``rank_bm25.BM25L``.

``rank_bm25``'s ``get_scores`` loops over every corpus document in Python for
each query term, which dominates the fresh-build cost of retrieval
preprocessing (minutes per pass on SciCite-scale corpora). This module keeps
the exact BM25L arithmetic

    idf(w)  = log(N + 1) - log(df(w) + 0.5)
    ctd     = tf(w, d) / (1 - b + b * |d| / avgdl)
    score   = sum_w idf(w) * tf * (k1 + 1) * (ctd + delta) / (k1 + ctd + delta)

but evaluates it with scipy CSC column slices, touching only the nonzero
term frequencies per query term. Scores agree with ``rank_bm25.BM25L`` to
float64 rounding; see ``scripts/bench_fast_bm25l.py``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse


class FastBM25L:
    def __init__(
        self,
        tok_docs: Sequence[List[str]],
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.corpus_size = len(tok_docs)

        doc_len = np.array([len(d) for d in tok_docs], dtype=np.float64)
        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean())

        vocab: Dict[str, int] = {}
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        df: Dict[str, int] = {}
        for d_idx, toks in enumerate(tok_docs):
            for term, cnt in Counter(toks).items():
                t_idx = vocab.setdefault(term, len(vocab))
                rows.append(d_idx)
                cols.append(t_idx)
                data.append(float(cnt))
                df[term] = df.get(term, 0) + 1

        self._vocab = vocab
        n_docs = self.corpus_size
        self.idf = {
            w: math.log(n_docs + 1) - math.log(n_df + 0.5)
            for w, n_df in df.items()
        }
        # CSC gives O(nnz) column slices per query term; keep raw arrays for
        # allocation-free scoring.
        self._tf = sparse.csc_matrix(
            (np.array(data), (rows, cols)),
            shape=(n_docs, len(vocab)),
        )
        self._indptr = self._tf.indptr
        self._indices = self._tf.indices
        self._data = self._tf.data
        len_norm = 1.0 - self.b + self.b * doc_len / self.avgdl
        self._inv_len_norm = 1.0 / len_norm

    # ------------------------------------------------------------------
    def get_scores(self, query: List[str]) -> np.ndarray:
        score = np.zeros(self.corpus_size, dtype=np.float64)
        indptr, indices, data = self._indptr, self._indices, self._data
        inv_len_norm = self._inv_len_norm
        k1, delta = self.k1, self.delta
        for term, q_count in Counter(query).items():
            t_idx = self._vocab.get(term)
            if t_idx is None:
                continue
            start, end = indptr[t_idx], indptr[t_idx + 1]
            if start == end:
                continue
            docs = indices[start:end]
            tf_wd = data[start:end]
            ctd = tf_wd * inv_len_norm[docs]
            contrib = self.idf[term] * tf_wd * (k1 + 1.0) * (ctd + delta) / (k1 + ctd + delta)
            score[docs] += q_count * contrib
        return score
