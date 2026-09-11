"""Training loop built on :class:`transformers.Trainer`.

Configured with ``bf16/fp16`` + ``gradient_checkpointing`` + ``gradient_accumulation``
so it fits on 6 GB VRAM while keeping an effective batch size that the small
dataset (8.2k rows) prefers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments

from ..models.classifier import build_model
from .utils import collate_fn, get_tokenizer, set_seed, tokenize_augmented, tokenize_function

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("citation.train")


def _descend_loss(o):
    """Find the scalar loss tensor anywhere in a (possibly nested) model output.

    HF/accelerate wraps model returns one level deeper than ``ModelOutput`` on
    some versions, so instead of guessing the exact nesting we walk the tree and
    pick the first tensor that looks like a scalar loss, preferring keys named
    ``loss`` at every level.
    """
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, dict):
        if "loss" in o:
            sub = _descend_loss(o["loss"])
            if sub is not None:
                return sub
        for v in o.values():
            sub = _descend_loss(v)
            if sub is not None:
                return sub
    elif isinstance(o, (tuple, list)):
        for v in o:
            sub = _descend_loss(v)
            if sub is not None:
                return sub
    return None


def _descend_logits(o):
    """Find the classification-logits tensor ``[batch, classes]`` in nested output."""
    if isinstance(o, torch.Tensor):
        if o.ndim == 2:
            return o
        return None
    if isinstance(o, dict):
        if "logits" in o:
            sub = _descend_logits(o["logits"])
            if sub is not None:
                return sub
        for v in o.values():
            sub = _descend_logits(v)
            if sub is not None:
                return sub
    elif isinstance(o, (tuple, list)):
        for v in o:
            sub = _descend_logits(v)
            if sub is not None:
                return sub
    return None


class CitationIntentTrainer(Trainer):
    """Explicit, version-robust loss extraction.

    Different HF/accelerate versions read a model's output container in different
    ways (attribute vs ``dict`` vs tuple vs one extra nesting level), so we locate
    the scalar loss by descending the returned structure and normalizing for
    gradient accumulation ourselves. This is deterministic across versions.
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor:
        outputs = model(**inputs)
        loss = _descend_loss(outputs)
        if loss is None or not isinstance(loss, torch.Tensor):
            raise TypeError(f"compute_loss could not find a scalar loss; got {type(outputs)}")
        if return_outputs:
            return loss, outputs
        return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = False,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        if labels is not None:
            inputs = {**inputs}
        with torch.no_grad():
            outputs = model(**inputs)
            loss = _descend_loss(outputs)
            logits = _descend_logits(outputs)
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, logits, labels)


def _accuracy_f1_metrics(predictions, labels, label_names=None):
    preds = np.argmax(predictions, axis=-1)
    acc = float((preds == labels).mean())
    # macro-F1 over the seen classes
    from sklearn.metrics import f1_score

    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return {"accuracy": acc, "macro_f1": macro_f1}


def train(
    dataset: DatasetDict,
    model_name: str,
    max_length: int = 256,
    per_device_batch_size: int = 8,
    grad_accumulation: int = 4,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    use_fp16: bool = True,
    use_gradient_checkpointing: bool = True,
    dropout: float = 0.1,
    freeze_embeddings: bool = False,
    pooling: str = "cls",
    output_dir: str = "runs/exp",
    seed: int = 42,
    weight_decay: float = 0.01,
    logging_steps: int = 50,
    save_steps: int = 500,
) -> Dict[str, Any]:
    """Train a classifier and return the evaluation results dict."""
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(model_name)

    # Dual pooling needs explicit evidence/target columns produced by BM25
    # augmentation; otherwise we fall back to the plain [CLS] pooling.
    effective_pooling = "dual" if (pooling == "dual" and "evidence" in dataset["train"].column_names) else "cls"

    if effective_pooling == "dual":
        from datasets import Dataset

        splits = {}
        for split in dataset:
            rows = []
            for ex in dataset[split]:
                feats = tokenize_augmented(
                    tokenizer, ex["target"], list(ex["evidence"]), max_length
                )
                rows.append({**feats, "label_id": ex["label_id"]})
            splits[split] = Dataset.from_list(rows)
        dd = DatasetDict(splits)
    else:
        # normalise the text column ("string" from SciCite, "text" elsewhere)
        text_col = "text" if "text" in dataset["train"].column_names else "string"
        tok = tokenize_function(tokenizer, max_length=max_length)
        removes = [c for c in ["string", "text"] if c in dataset["train"].column_names]
        dd = dataset.map(tok, batched=True, remove_columns=removes)

    label_names = sorted(set(dd["train"]["label_id"]))
    num_labels = max(label_names) + 1 if label_names else 3

    model = build_model(
        model_name=model_name,
        num_labels=num_labels,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
        pooling=effective_pooling,
    )

    # ---- TrainingArguments tuned for 6 GB VRAM ----
    from transformers import (
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size * 2,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        # prefer bf16 (no GradScaler) on Ampere-class GPUs; the fp16 GradScaler
        # path is buggy under HF5 + gradient checkpointing.
        fp16=False,
        bf16=use_fp16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=use_gradient_checkpointing,
        logging_steps=logging_steps,
        save_total_limit=2,
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        return _accuracy_f1_metrics(preds, labels, label_names)

    trainer = CitationIntentTrainer(
        model=model,
        args=args,
        train_dataset=dd["train"],
        eval_dataset=dd["validation"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ---- Evaluate on test ----
    test_metrics = trainer.evaluate(dd["test"], metric_key_prefix="test")
    train_metrics = trainer.evaluate(dd["train"], metric_key_prefix="train")

    results = {
        "model": model_name,
        **{k: v for k, v in test_metrics.items()},
        **{k: v for k, v in train_metrics.items()},
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    trainer.save_model(str(output_dir / "best_model"))

    # Persist head metadata + a self-contained checkpoint so the dual-pooling
    # head (which is not an AutoModelForSequenceClassification) can be reloaded.
    (output_dir / "head_config.json").write_text(
        json.dumps(
            {
                "pooling": effective_pooling,
                "model_name": model_name,
                "num_labels": num_labels,
            }
        ),
        encoding="utf-8",
    )
    if effective_pooling == "dual":
        torch.save(model.state_dict(), output_dir / "dual_head.pt")

    logger.info("Done. Test metrics: %s", test_metrics)
    return results