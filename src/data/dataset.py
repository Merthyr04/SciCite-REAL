"""Unified data loading & normalization for the SciCite citation-intent dataset.

The official SciCite release (Cohan et al., NAACL 2019) is a JSONL file where each
line is one annotated citation. We normalise it into a huggingface ``Dataset`` with
consistent column names so the downstream pipeline is dataset-agnostic:

    string            text of the citing paragraph that contains the citation
    sectionName       section of the citing paper where the citation appears
    label             primary (3-class) intent: background | method | result
    label2            optional fine-grained (4-class) intent, when present
    source            provenance of the annotation
    citingPaperId     id of the citing paper
    citedPaperId      id of the cited paper
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict

# Ordered, stable label id for the primary 3-class task.
PRIMARY_LABELS: List[str] = ["background", "method", "result"]
PRIMARY_LABEL2ID: Dict[str, int] = {lbl: i for i, lbl in enumerate(PRIMARY_LABELS)}
PRIMARY_ID2LABEL: Dict[int, str] = {i: lbl for lbl, i in PRIMARY_LABEL2ID.items()}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a SciCite-format JSONL file into a list of dicts.""" 
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the fields we need and coerce types consistently."""
    label = row.get("label", "background")  # primary intent string
    label2 = row.get("label2")

    def _s(v) -> str:
        return "" if v is None else str(v)

    return {
        "string": _s(row.get("context", row.get("string", ""))).strip(),
        "sectionName": _s(row.get("sectionName", row.get("section", ""))),
        "label": str(label).strip(),
        "label2": (str(label2).strip() if label2 is not None else None),
        "source": _s(row.get("source")),
        "citingPaperId": _s(row.get("citingPaperId")),
        "citedPaperId": _s(row.get("citedPaperId")),
        "id": _s(row.get("id", row.get("excerpt_index", ""))),
    }


def load_scicite(
    data_dir: str | Path,
    use_label2: bool = False,
) -> DatasetDict:
    """Load SciCite train / validation / test splits.

    Args:
        data_dir: directory containing ``train.jsonl``, ``dev.jsonl`` (or
            ``val.jsonl``), ``test.jsonl``.
        use_label2: if True, use the fine-grained 4-class annotation as the target.

    Returns:
        A DatasetDict with train / validation / test splits; the ``label`` column
        holds the resolved intent (3-class by default, 4-class when requested).
    """
    data_dir = Path(data_dir)

    def _key(*names: str) -> str:
        for n in names:
            if (data_dir / n).exists():
                return n
        raise FileNotFoundError(
            f"None of {[str(n) for n in names]} found under {data_dir}"
        )

    train_file = _key("train.jsonl")
    dev_file = _key("dev.jsonl", "valid.jsonl", "validation.jsonl")
    test_file = _key("test.jsonl")

    splits = {}
    for split, fname in [("train", train_file), ("validation", dev_file), ("test", test_file)]:
        rows = [_normalise_row(r) for r in read_jsonl(data_dir / fname)]
        splits[split] = Dataset.from_list(rows)

    dd = DatasetDict(splits)

    # Resolve the target label column.
    def _resolve(row: Dict) -> Dict:
        raw = row["label2"] if use_label2 else row["label"]
        row["label"] = raw if raw is not None else "unknown"
        return row

    dd = dd.map(_resolve)

    label_names = sorted(set(dd["train"]["label"]) | {"background", "method", "result"})
    ddict = {lbl: i for i, lbl in enumerate(label_names)}
    dd = dd.map(lambda r: {"label_id": ddict[r["label"]]})
    return dd


def stat_summary(dd: DatasetDict) -> str:
    """Return a short human-readable summary of split sizes and label balance."""
    out = []
    for split in dd:
        n = len(dd[split])
        counts = {lbl: 0 for lbl in set(dd["train"]["label"])}
        for lbl in dd[split]["label"]:
            counts[lbl] = counts.get(lbl, 0) + 1
        out.append(f"{split:>10}: {n:>6}  {counts}")
    return "\n".join(out)