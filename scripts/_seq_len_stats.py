"""Sequence-length distribution for baseline vs augmented inputs.

Establishes what padding each configuration actually pays for, so the
efficiency table in the paper can state the comparison honestly.
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"e:\Paper\SciCite-REAL")

from src.data.dataset import load_scicite
from src.models.classifier import resolve_model_name
from src.training.utils import tokenize_augmented

ROOT = Path(r"e:\Paper\SciCite-REAL")
MAX_LEN = 256

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(resolve_model_name("deberta"))

dd = load_scicite(str(ROOT / "data" / "processed" / "scicite"))

# baseline: raw target strings, fixed max_length=256 padding
targets = dd["train"]["string"]
base_lens = []
for s in targets:
    ids = tok(s, truncation=True, max_length=MAX_LEN)["input_ids"]
    base_lens.append(len(ids))

# augmented: reuse the cached evidence for topk3/leak0
cache_dir = (ROOT / "data" / "processed" / "cache").resolve()
aug_files = sorted(cache_dir.glob("augment_*.pkl"))
print("cache files:", [f.name for f in aug_files])
aug = pickle.load(open(aug_files[0], "rb"))  # dict split -> dict of lists
print("cache keys:", list(aug.keys()), "| columns:", list(aug["train"].keys()))

aug_lens = []
n = len(aug["train"]["target"])
for i in range(n):
    feats = tokenize_augmented(
        tok, aug["train"]["target"][i], list(aug["train"]["evidence"][i]), MAX_LEN
    )
    aug_lens.append(len(feats["input_ids"]))


def stats(name, lens, fixed=None):
    lens = sorted(lens)
    n = len(lens)
    mean = sum(lens) / n
    p50 = lens[n // 2]
    p90 = lens[int(n * 0.9)]
    p99 = lens[int(n * 0.99)]
    tot = sum(lens)
    if fixed:
        pad = fixed * n - tot
        print(f"{name}: n={n} mean={mean:.1f} p50={p50} p90={p90} p99={p99} "
              f"max={lens[-1]} | tokens={tot:,} | padded-to-{fixed} waste={pad:,} "
              f"({pad / (fixed * n) * 100:.0f}% of padded)")
    else:
        print(f"{name}: n={n} mean={mean:.1f} p50={p50} p90={p90} p99={p99} max={lens[-1]} | tokens={tot:,}")
    return tot


print()
tot_base = stats("baseline target", base_lens)
print(f"baseline as-trained (fixed 256): tokens={256 * len(base_lens):,}")
tot_aug = stats("augmented (dynamic)", aug_lens)
print()
print(f"augmented/baseline token ratio (raw): {tot_aug / tot_base:.2f}x")
print(f"augmented tokens vs fixed-256 padded: {tot_aug / (256 * len(base_lens)):.2f}x")
