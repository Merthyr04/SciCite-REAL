"""BM25 + BM25L index over a pool of citation contexts.

``BM25Index`` wraps ``rank_bm25`` with the L-weighting variant (BM25L, Lv &
Zhai, 2011), which is noticeably better than classic BM25 when documents have
very unequal lengths — precisely the case for citation snippets of 5 vs 60 words.
Copy-tolerant term frequency scoring is handled by rank_bm25's parameters.

The index supports a per-query ``exclude`` set so that:
  * we never retrieve the target context itself, and
  * for leak-free ablations we can forbid retrieving contexts that carry the
    same label (i.e. retrieval must provide genuinely *different* evidence).
"""

from __future__ import annotations

import heapq
from typing import Dict, Iterable, List, Optional, Sequence, Set

from rank_bm25 import BM25L, BM25Okapi

from ..data.preprocessing import tokenize_for_bm25
from .fast_bm25l import FastBM25L


class BM25Index:
    """A small, dependency-light BM25 index over text passages."""

    variant: str

    def __init__(
        self,
        documents: Sequence[str],
        variant: str = "bm25l",
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.5,
    ) -> None:
        self.corpus = list(documents)
        if not self.corpus:
            raise ValueError("BM25Index requires at least one document")
        # O(1) lookup of corpus positions by their exact text, so per-query
        # self-exclusion never scans the whole corpus.
        self._text2idx: Dict[str, List[int]] = {}
        for i, d in enumerate(self.corpus):
            self._text2idx.setdefault(d, []).append(i)
        tok_docs = [tokenize_for_bm25(d) for d in self.corpus]
        if variant == "bm25l":
            # vectorised scorer; scores match rank_bm25.BM25L to float rounding
            self._model = FastBM25L(tok_docs, k1=k1, b=b, delta=delta)
            self.variant = "bm25l"
        elif variant == "bm25":
            self._model = BM25Okapi(tok_docs, k1=k1, b=b)
            self.variant = "bm25"
        else:
            raise ValueError(f"Unknown BM25 variant: {variant!r}")

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        exclude_ids: Optional[Set[int]] = None,
    ) -> List[Dict]:
        """Return ``top_k`` scored hits as ``{idx, score, text}``.

        Args:
            query: the citation context we want evidence for.
            top_k: number of similar contexts to return.
            exclude_ids: document indices that must never be returned.
        """
        scores = self._model.get_scores(tokenize_for_bm25(query))
        if exclude_ids:
            for i in exclude_ids:
                scores[i] = float("-inf")
        # only the top-k matter; a heap is O(n log k) instead of a full sort.
        top = heapq.nlargest(top_k + (len(exclude_ids) if exclude_ids else 0),
                             range(len(scores)),
                             key=scores.__getitem__)
        hits = []
        for i in top:
            if scores[i] == float("-inf"):
                continue
            hits.append(
                {"idx": int(i), "score": float(scores[i]), "text": self.corpus[i]}
            )
            if len(hits) >= top_k:
                break
        return hits


def build_index_from_df(df, text_col: str = "string") -> BM25Index:
    """Convenience builder from a pandas/Dataset-like frame."""
    return BM25Index(list(df[text_col]))