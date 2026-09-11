"""Encoder + classification-head model factory.

The same backbone is shared by all baselines and the retrieval-augmented model;
the only difference is the *input text* fed to it. This keeps comparisons fair:
any accuracy gap must come from the extra evidence, not from a different head.

For GPU-memory-constrained hardware (e.g. a 6 GB RTX 3060) the following flags
are exposed so a full-size backbone can still be fine-tuned:

  * ``gradient_checkpointing=True`` trades compute for ~40% activation memory.
  * ``freeze_embeddings=True`` keeps the (very large) embedding matrix fixed.
  * ``dropout`` can be raised for the small-data regime (~8.2k training rows).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional
from torch import Tensor
from transformers import AutoConfig, AutoModel

MODEL_ALIASES = {
    "bert": "e:/Paper/SciCite-REAL/data/models/bert",
    "scibert": "e:/Paper/SciCite-REAL/data/models/scibert",
    "deberta": "e:/Paper/SciCite-REAL/data/models/deberta",
    "deberta-small": "e:/Paper/SciCite-REAL/data/models/deberta",
}


class ModelOutput(dict):
    """Minimal attribute/dict-access container for the model returns.

    Trainer (and ``torch.compile`` / gradient checkpointing) access outputs both
    via attributes (``output.loss``) and as dicts (``output["logits"]``), so we
    back it with a plain dict that also exposes keys as attributes.
    """

    def __init__(self, loss=None, logits=None, hidden_states=None):
        super().__init__()
        self["loss"] = loss
        self["logits"] = logits
        self["hidden_states"] = hidden_states

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def resolve_model_name(name: str) -> str:
    """Resolve a short alias to the local cached checkpoint.

    The three backbones are downloaded once (see ``scripts/dl_*.py``) and cached
    locally; resolution returns the local path so training never re-downloads.
    """
    return MODEL_ALIASES.get(name, name)


class CitationIntentClassifier(nn.Module):
    """Minimal, explicit head for transparency (equivalent to AutoModelForSC)."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        freeze_embeddings: bool = False,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        resolved = resolve_model_name(model_name)
        config = AutoConfig.from_pretrained(resolved, num_labels=num_labels)
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        # The cached weights are stored half precision; force fp32 so the
        # freshly-initialised fp32 head and the backbone have one consistent
        # dtype. Mixed-precision params silently collapse to the majority class.
        self.backbone = AutoModel.from_pretrained(
            resolved, config=config, torch_dtype=torch.float32
        ).float()
        # BERT-class models tie the input/output embeddings, so `.float()`
        # yields *views* of shared storage that bt training-time safe-serializers
        # reject as non-contiguous. Materialise every parameter once.
        for _p in self.backbone.parameters():
            if not _p.is_contiguous():
                _p.data = _p.data.contiguous()
        if freeze_embeddings:
            self.backbone.embeddings.float()

        hidden = config.hidden_size
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        if pooling == "dual":
            # Evidence-aware head: gate fuses the sentence-level [CLS] vector with
            # the mean pooling over the *evidence* segments (segment_ids == 1).
            self.classifier = nn.Linear(hidden, num_labels)
            self.head_gate = nn.Linear(2 * hidden, hidden)
        else:
            self.classifier = nn.Linear(hidden, num_labels)
            self.head_gate = None

        if freeze_embeddings:
            print("[config] Freezing embedding matrix.")
            for p in self.backbone.embeddings.parameters():
                p.requires_grad = False

    def gradient_checkpointing_enable(self, *args, **kwargs) -> None:
        """Delegate to the backbone so Trainer can toggle checkpointing."""
        self.backbone.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def resize_token_embeddings(self, *a, **k):
        return self.backbone.resize_token_embeddings(*a, **k)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        segment_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = out.last_hidden_state  # [B, L, H]

        if self.pooling == "dual" and segment_ids is not None:
            cls_pool = hidden[:, 0]
            ev_mask = (segment_ids == 1).long()  # [B, L]
            denom = ev_mask.sum(dim=1, keepdim=True).clamp(min=1)
            ev_pool = (hidden * ev_mask.unsqueeze(-1)).sum(dim=1) / denom
            # learnable gate on [cls; evidence]
            g = torch.sigmoid(self.head_gate(torch.cat([cls_pool, ev_pool], dim=-1)))
            fused = g * cls_pool + (1.0 - g) * ev_pool
            logits = self.classifier(self.dropout(fused))
        else:
            pooler = hidden[:, 0]
            logits = self.classifier(self.dropout(pooler))

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return ModelOutput(
            loss=loss, logits=logits, hidden_states=out.hidden_states
        )


def build_model(
    model_name: str,
    num_labels: int,
    dropout: float = 0.1,
    freeze_embeddings: bool = False,
    pooling: str = "cls",
) -> nn.Module:
    return CitationIntentClassifier(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
        pooling=pooling,
    )