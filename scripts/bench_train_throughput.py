"""Controlled training-throughput benchmark: baseline vs retrieval-augmented.

Reproduces the exact training configuration of the reported runs
(batch 8, grad-accum 4, bf16 autocast, gradient checkpointing, fp32
backbone load) and measures steady-state training throughput for:

  1. baseline  - plain DeBERTa-v3 + CLS head, fixed max_length=256 padding
  2. dual      - DeBERTa-v3 + gated dual-pooling head on cached k=3 BM25L
                 evidence, dynamic per-batch padding (as the real runs did)

Writes runs/train_throughput_bench.json.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(r"e:\Paper\SciCite-REAL")
sys.path.insert(0, str(ROOT))

import src.data.dataset as dsd
from src.models.classifier import build_model
from src.training.utils import collate_fn, get_tokenizer, set_seed, tokenize_augmented, tokenize_function

MAX_LEN = 256
BATCH = 8
ACCUM = 4
WARMUP_STEPS = 4
TIMED_STEPS = 30


def timed_run(model_name: str, pooling: str, examples) -> dict:
    set_seed(42)
    model = build_model(model_name=model_name, num_labels=3, dropout=0.1,
                        freeze_embeddings=False, pooling=pooling)
    model.gradient_checkpointing_enable()
    model.to("cuda")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # flat list of collated batches (dynamic padding per batch)
    g = torch.Generator().manual_seed(42)
    order = torch.randperm(len(examples), generator=g).tolist()
    bat = []
    for i in range(0, len(order) - BATCH + 1, BATCH):
        rows = [examples[j] for j in order[i:i + BATCH]]
        bat.append(collate_fn(rows))

    micro_total = (WARMUP_STEPS + TIMED_STEPS) * ACCUM
    micro_done = 0
    t_start = None
    opt.zero_grad(set_to_none=True)
    while micro_done < micro_total:
        b = bat[micro_done % len(bat)]
        b = {k: v.to("cuda", non_blocking=True) for k, v in b.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(**b)
        (out["loss"] / ACCUM).backward()
        micro_done += 1
        if micro_done % ACCUM == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if micro_done == WARMUP_STEPS * ACCUM:
            torch.cuda.synchronize()
            t_start = time.perf_counter()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    samples = TIMED_STEPS * ACCUM * BATCH
    del model, opt, bat
    torch.cuda.empty_cache()
    return {"samples_per_second": round(samples / elapsed, 1),
            "timed_optimizer_steps": TIMED_STEPS,
            "elapsed_s": round(elapsed, 2),
            "n_examples": len(examples)}


def main():
    tok = get_tokenizer("deberta")

    # ---- baseline: fixed 256 padding, plain text ----
    dd = dsd.load_scicite(str(ROOT / "data" / "processed" / "scicite"))
    tokf = tokenize_function(tok, max_length=MAX_LEN)
    base_examples = []
    for ex in dd["train"]:
        enc = tok([ex["string"]], padding="max_length", truncation=True,
                  max_length=MAX_LEN)
        base_examples.append({"input_ids": enc["input_ids"][0],
                              "attention_mask": enc["attention_mask"][0],
                              "label_id": ex["label_id"]})

    # ---- dual: cached k=3 / leak-0 evidence, dynamic padding ----
    cache_key = hashlib.sha1(
        b"augment_topk3_leak0_srcdata/index/scicite_bm25l_train.pkl"
    ).hexdigest()[:12]
    cache_path = (ROOT / "data" / "processed" / "cache" / f"augment_{cache_key}.pkl").resolve()
    print("cache:", cache_path.name, "exists:", cache_path.exists())
    aug = pickle.load(open(cache_path, "rb"))
    dual_examples = []
    for i in range(len(aug["train"]["target"])):
        feats = tokenize_augmented(tok, aug["train"]["target"][i],
                                   list(aug["train"]["evidence"][i]), MAX_LEN)
        dual_examples.append({**feats, "label_id": aug["train"]["label_id"][i]})

    print("baseline throughput ...")
    base = timed_run("deberta", "cls", base_examples)
    print(json.dumps(base))
    print("dual throughput ...")
    dual = timed_run("deberta", "dual", dual_examples)
    print(json.dumps(dual))

    results = {
        "config": {"model": "deberta-v3-base", "batch": BATCH, "accum": ACCUM,
                   "bf16_autocast": True, "gradient_checkpointing": True,
                   "max_length": MAX_LEN, "warmup_steps": WARMUP_STEPS,
                   "timed_steps": TIMED_STEPS},
        "baseline_fixed256": base,
        "dual_dynamic_k3": dual,
        "overhead_pct": round((base["samples_per_second"] - dual["samples_per_second"])
                              / base["samples_per_second"] * 100, 1),
    }
    out = ROOT / "runs" / "train_throughput_bench.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
