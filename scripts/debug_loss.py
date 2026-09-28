"""Isolate whether gradient_checkpointing / fp16 / cuda causes nested outputs."""
import os, sys
os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.classifier import build_model

def probe(label, fn):
    out = fn()
    name = type(out).__name__
    inner = out["loss"] if isinstance(out, dict) else None
    inner_t = type(inner).__name__
    nested = "NESTED" if isinstance(inner, dict) else "flat"
    print(f"{label:38s} out={name:12s} loss->{inner_t:10s} {nested}")
    return out

m = build_model(model_name="bert", num_labels=3, pooling="cls").cuda()
ids = torch.randint(0, 1000, (2, 32)); mask = torch.ones_like(ids); labels = torch.tensor([0, 1])
ids, mask, labels = ids.cuda(), mask.cuda(), labels.cuda()

probe("cuda, no ckpt, no autocast", lambda: m(input_ids=ids, attention_mask=mask, labels=labels))
m.train()

m2 = build_model(model_name="bert", num_labels=3, pooling="cls").cuda()
m2.gradient_checkpointing_enable()
probe("cuda + gradient_checkpointing", lambda: m2(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()))

m3 = build_model(model_name="bert", num_labels=3, pooling="cls").cuda()
with torch.autocast("cuda", dtype=torch.float16):
    probe("cuda + fp16 autocast", lambda: m3(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()))

m4 = build_model(model_name="bert", num_labels=3, pooling="cls").cuda()
m4.gradient_checkpointing_enable()
with torch.autocast("cuda", dtype=torch.float16):
    probe("cuda + ckpt + fp16", lambda: m4(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()))