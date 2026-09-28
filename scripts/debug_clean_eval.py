"""Clean, Trainer-free evaluation of the trained dual model on the test split.

HF Trainer's eval path returns nested/interleaved logits on transformers 5.17,
so we compute paper numbers here with a plain forward pass instead.
"""
import os, sys, argparse, pickle
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.data.dataset import load_scicite, PRIMARY_LABELS
from src.retrieval.retrieval_augmented import RetrievalAugmenter
from src.models.classifier import build_model
from src.training.utils import get_tokenizer, tokenize_augmented

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = os.path.join(root, "data/processed/scicite")
index_path = os.path.join(root, "data/index/scicite_bm25l_train.pkl")
pooling = "dual"

dd = load_scicite(data)
with open(index_path, "rb") as fh:
    index = pickle.load(fh)
aug = RetrievalAugmenter(index=index, top_k=3, evidence_source="train",
                         leak_free=False, labels=PRIMARY_LABELS)

# tokenize all test instances with evidence + segment ids
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
tok = get_tokenizer("deberta")
n_test = len(dd["test"]) if a.limit is None else min(a.limit, len(dd["test"]))
encs, label_ids = [], []
for i in range(n_test):
    ex = dd["test"][i]
    a_ = aug.augment(target_text=ex["string"], target_label=ex["label"], doc_id=ex["id"])
    enc = tokenize_augmented(tok, a_["target"], a_["evidence"], 256)
    encs.append(enc)
    label_ids.append(ex["label_id"])

def pad(key):
    seqs = [e[key] for e in encs]; m = max(len(s) for s in seqs)
    return torch.tensor([s + [0]*(m-len(s)) for s in seqs], dtype=torch.long)

input_ids, attention_mask, segment_ids = pad("input_ids"), pad("attention_mask"), pad("segment_ids")
true = np.array(label_ids)

ckpt = os.path.join(root, "runs/real_deberta_dual_k3/dual_head.pt")
model = build_model(model_name="deberta", num_labels=3, pooling=pooling)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model = model.float().eval()

with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask, segment_ids=segment_ids)
logits = out.logits if hasattr(out, "logits") else None
if isinstance(logits, dict):
    logits = (logits or {}).get("logits")
preds = logits.argmax(-1).numpy()

from collections import Counter
print("true dist:", dict(Counter(true.tolist())))
print("pred dist:", dict(Counter(preds.tolist())))
print("accuracy:", float((preds == true).mean()))
print("macro_f1 :", float(f1_score(true, preds, average="macro", zero_division=0)))