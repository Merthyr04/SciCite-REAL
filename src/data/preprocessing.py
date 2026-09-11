"""Lightweight text preprocessing shared by retrieval indexing and tokenisation.

For retrieval (BM25) we want aggressive, recall-friendly normalisation:
whitespace folding, punctuation stripping and lowercasing. The transformer
tokeniser does its own sub-word tokenisation afterwards, so we keep the raw
sentence for the model input and only compute the BM25 query/document tokens here.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List

_PUNCT_RE = re.compile(r"[^\w\s]")


@lru_cache(maxsize=4096)
def normalise(text: str) -> str:
    """Lowercase, strip punctuation and fold whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_for_bm25(text: str) -> List[str]:
    """Split normalised text on whitespace; returns BM25-indexable tokens."""
    return normalise(text).split()


def tokenize_for_bm25_batch(texts: List[str]) -> List[List[str]]:
    return [tokenize_for_bm25(t) for t in texts]