"""Time the cached re-build path: load the augment pickle as run_training does."""
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(r"e:\Paper\SciCite-REAL")
sys.path.insert(0, str(ROOT))

from datasets import Dataset, DatasetDict

t0 = time.perf_counter()
with open(ROOT / "data" / "processed" / "cache" / "augment_f8696d53c362.pkl", "rb") as fh:
    raw = pickle.load(fh)
dd = DatasetDict({k: Dataset.from_dict(v) for k, v in raw.items()})
t1 = time.perf_counter()
print(f"cached re-build (pickle -> DatasetDict): {t1 - t0:.2f}s")
print("splits:", {k: len(v) for k, v in dd.items()})
