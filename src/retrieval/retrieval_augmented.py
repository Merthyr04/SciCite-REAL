"""Retrieval-augmented input construction.

Given a target citation context, we retrieve ``top_k`` similar contexts from a
pool (configurable) and concatenate them in front of the target to form the
model input:

    [CLS] ret<1> [SEP] ret<2> [SEP] ... ret<k> [SEP] target [SEP]

The retrieved passages act as *evidence*: repeated phrases and construction
patterns that co-occur with a given citation intent. Two hyper-parameters
control the evidence source and its risk of shortcutting:

  * ``evidence_source`` :
      - "corpus"  — pool = all contexts (default)
      - "train"   — pool = training contexts only (recommended for the test set,
                     avoids using the test set itself during retrieval)
      - "expand"  — pool = contexts from the same section of the target
  * ``leak_free`` :
      True forbids retrieving contexts whose label equals the target's, forcing
      the model to use *structural* evidence rather than copying a label.

```rst
NOTE ON LEAKAGE
Retrieval augmentation on the *test* split must NOT retrieve test contexts
other than via a train-only pool, otherwise the retrieved phrase may contain
the very answer. Our default protocol is therefore:
   * train retrieval pool -> train-set contexts (excluding self)
   * eval  retrieval pool -> train-set contexts
   * when leak_free=True -> additionally drop same-label train contexts
```
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Set

from .index import BM25Index


class RetrievalAugmenter:
    def __init__(
        self,
        index: Optional[BM25Index] = None,
        top_k: int = 3,
        evidence_source: str = "train",
        leak_free: bool = False,
        labels: Optional[Sequence[str]] = None,
        sep: str = " ",
    ) -> None:
        self.index = index
        self.top_k = top_k
        self.evidence_source = evidence_source
        self.leak_free = leak_free
        self.sep = sep
        self._labels = list(labels) if labels is not None else None

    # ------------------------------------------------------------------
    def set_index(self, index: BM25Index) -> None:
        self.index = index

    def _exclude_self(self, doc_id: str) -> Set[int]:
        """Return corpus positions whose text equals the target (self-match)."""
        exclude: Set[int] = set()
        if self.index is None:
            return exclude
        mapping = getattr(self.index, "_text2idx", None)
        if mapping is None:  # tolerate indexes pickled by the pre-map code
            mapping = {}
            for i, d in enumerate(self.index.corpus):
                mapping.setdefault(d, []).append(i)
            self.index._text2idx = mapping
        for i in mapping.get(doc_id, ()):
            exclude.add(i)
        return exclude

    # ------------------------------------------------------------------
    def augment(
        self,
        target_text: str,
        target_label: Optional[str] = None,
        doc_id: str = "",
    ) -> Dict:
        """Return ``{"target": ..., "evidence": [..texts..]}`` for one instance."""
        if self.index is None:
            return {"text": target_text, "evidence": []}

        exclude = self._exclude_self(target_text)

        # Leak-free mode: the index may carry per-doc labels (set in the build
        # script). Forbid retrieving contexts that share the target's label, so
        # the model cannot "copy" the answer and must rely on structural evidence.
        if self.leak_free and target_label is not None:
            idx_labels = getattr(self.index, "labels", None)
            if idx_labels is not None:
                for i, lbl in enumerate(idx_labels):
                    if str(lbl) == str(target_label):
                        exclude.add(i)

        hits = self.index.retrieve(target_text, top_k=self.top_k, exclude_ids=exclude)
        evidence = [h["text"] for h in hits]
        input_text = self.sep.join(evidence + [target_text])
        return {"text": input_text, "evidence": evidence, "target": target_text}


def batch_augment(
    augmenter: RetrievalAugmenter,
    instances: List[Dict],
) -> List[Dict]:
    """Apply augmentation to a list of ``{"text","label_id",...}`` instances."""
    out = []
    for inst in instances:
        a = augmenter.augment(
            target_text=inst["string"],
            target_label=inst.get("label"),
            doc_id=inst.get("id", ""),
        )
        out.append({**inst, **a})
    return out