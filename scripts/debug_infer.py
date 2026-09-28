"""Decide whether the trained model learned (diverse logits) or collapsed."""
import os, sys
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_from_disk
from src.models.classifier import build_model
from src.training.utils import get_tokenizer, tokenize_augmented, collate_fn
from src.data.dataset import load_scicite

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dd = load_scicite(os.path.join(root, "data/processed/scicite"))

# rebuild dual model from saved state_dict
model = build_model(model_name="deberta", num_labels=3, pooling="dual")
state = torch.load(os.path.join(root, "runs/real_deberta_dual_k3/dual_head.pt"), map_location="cpu")
model.load_state_dict(state)
model.eval()

tok = get_tokenizer("deberta")
n = 40
rows, labels = [], []
for i in range(n):
    ex = dd["train"][i]
    rows.append(tokenize_augmented(tok, ex["string"], list(ex["evidence"]), 256))
    labels.append(ex["label_id"])
batch = collate_fn([{**r, "label_id": l} for r, l in zip(rows, labels)])
with torch.no_grad():
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                segment_ids=batch.get("segment_ids"), labels=batch["labels"])
logits = out.logits if hasattr(out, "logits") else None
if isinstance(logits, dict):
    logits = logits["logits"] if isinstance(logits.get("logits"), torch.Tensor) else None
preds = logits.argmax(-1).numpy() if logits is not None else None
print("true  labels:", labels)
print("pred  labels:", list(preds) if preds is not None else None)
if logits is not None:
    from collections import Counter
    print("pred distribution:", Counter(preds.tolist()))
    import numpy as np
    print("logits probs sample:", torch.softmax(logits[:5], -1).numpy().round(3).tolist())
    print("acc:", float((preds == np.array(labels)).mean()))