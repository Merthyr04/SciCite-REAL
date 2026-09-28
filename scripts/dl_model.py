#!/usr/bin/env python3
"""Download HF model weights via a fast source (ModelScope first, HF fallback).

ModelScope's CDN is typically much faster than huggingface.co from China. We try
in order: (1) ModelScope direct, (2) hf-mirror.com, (3) huggingface.co.

Usage:
    python scripts/dl_model.py --model bert --out data/models/bert
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

# (model alias) -> list of (kind, url) sources, checked in order
SOURCES = {
    "bert": [
        ("ms", "https://www.modelscope.cn/models/AI-ModelScope/bert-base-uncased/resolve/master/pytorch_model.bin"),
        ("hf", "https://hf-mirror.com/bert-base-uncased/resolve/main/pytorch_model.bin"),
        ("hf", "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"),
    ],
}

CONFIG_URLS = {
    "bert": [
        ("ms", "https://www.modelscope.cn/models/AI-ModelScope/bert-base-uncased/resolve/master/config.json"),
        ("hf", "https://hf-mirror.com/bert-base-uncased/resolve/main/config.json"),
    ],
}


def download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        total = int(r.headers.get("Content-Length", 0) or 0)
        with open(dest, "wb") as f:
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                f.write(b)
    print(f"  dl ok {dest} ({round(total / 1e6, 1)} MB, {round((time.time() - t0) / 60, 1)} min)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", action="store_true", help="also fetch vocab/tokenizer files")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for kind, url in SOURCES.get(args.model, []):
        try:
            print(f"[try {kind}] {url}")
            if download(url, out / "pytorch_model.bin"):
                break
        except Exception as e:
            print(f"  {kind} failed: {str(e)[:80]}")
    else:
        sys.exit("ALL SOURCES FAILED")


if __name__ == "__main__":
    main()