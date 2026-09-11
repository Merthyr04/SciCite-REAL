"""Shared training utilities: tokenisation, collator, metrics, IO helpers."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..models.classifier import resolve_model_name


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(resolve_model_name(model_name))


def tokenize_function(
    tokenizer,
    max_length: int = 256,
    pad_to_multiple_of: Optional[int] = None,
):
    """Return a callable that maps a batch dict with ``text`` -> input tensors."""

    def _fn(batch: Dict[str, List]) -> Dict:
        enc = tokenizer(
            batch["text"],
            padding="max_length" if pad_to_multiple_of else True,
            # use "max_length" style fixed length for deterministic tensor shapes
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        }

    return _fn


def tokenize_augmented(
    tokenizer,
    target: str,
    evidence: List[str],
    max_length: int = 256,
) -> Dict[str, List[int]]:
    """Segment-aware tokenisation for the dual-pooling head.

    Returns ``input_ids``, ``attention_mask`` and ``segment_ids`` where segments
    are 0 for the target context (and the [CLS]/separators around it) and 1 for
    every token belonging to a retrieved evidence sentence. Keeps the target
    intact under truncation and instead drops evidence tokens first, since the
    label signal lives almost entirely in the target context.
    """
    sep = tokenizer.sep_token_id
    cls_ = tokenizer.cls_token_id
    budget = max_length - 2  # reserve [CLS] and the terminal [SEP]

    # tokenise evidence with a per-sentence cap so one long neighbour cannot hog
    # the whole budget when k > 1
    ev_inputs = []
    per_cap = max(16, budget // max(1, len(evidence)))
    for ev in evidence:
        ids = tokenizer(ev, add_special_tokens=False, truncation=True,
                        max_length=per_cap)["input_ids"]
        ev_inputs.append(ids)

    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

    # trim evidence from the oldest sentence first until everything fits
    while sum(len(e) + 1 for e in ev_inputs) + len(target_ids) > budget and ev_inputs:
        ev_inputs[0] = ev_inputs[0][:-1]
        if not ev_inputs[0]:
            ev_inputs.pop(0)

    input_ids = [cls_]
    segment_ids = [0]
    for ids in ev_inputs:
        input_ids += ids
        segment_ids += [1] * len(ids)
        input_ids.append(sep)
        segment_ids.append(1)
    input_ids += target_ids
    segment_ids += [0] * len(target_ids)
    input_ids.append(sep)
    segment_ids.append(0)

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": segment_ids,
    }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Pad a ragged batch of already-tokenised examples into tensors + labels.

    When examples carry ``segment_ids`` (dual-pooling head), they are padded the
    same way as ``input_ids`` and returned as well. Padding is applied to the
    *dynamic* sequence length of a batch so that neither train nor eval require
    a fixed ``max_length`` budget.
    """

    def _pad(key: str) -> torch.Tensor:
        seqs = [b[key] for b in batch]
        maxlen = max(len(s) for s in seqs)
        padded = [s + [0] * (maxlen - len(s)) for s in seqs]
        return torch.tensor(padded, dtype=torch.long)

    input_ids = _pad("input_ids")
    attn = _pad("attention_mask")
    labels = torch.tensor([b["label_id"] for b in batch], dtype=torch.long)
    out = {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    if "segment_ids" in batch[0]:
        out["segment_ids"] = _pad("segment_ids")
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)