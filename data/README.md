# Data notes

This directory holds the **derived** data released with SciCite-REAL. The raw
corpora are not redistributed here; `scripts/download_data.py` re-fetches them.

## Licensing

- **Code** in this repository is released under the Apache License 2.0 — see the
  top-level `LICENSE`.
- **SciCite** (Cohan et al., NAACL 2019) is released by the Allen Institute for
  AI under Apache-2.0. The authoritative source is
  <https://github.com/allenai/SciCite>; consult it for licensing and citation
  details.
- **ACL-ARC** (Jurgens et al., 2018) is redistributed under the terms of its
  original release.
- Everything in this directory is a **derived artifact** (preprocessed splits,
  retrieval indices, augmented inputs) produced from those corpora, and is
  released under the same Apache-2.0 terms as the code.

## Layout

```
data/
├── index/                      # BM25 / BM25L retrieval indices (pickles)
│   ├── scicite_bm25_train.pkl
│   ├── scicite_bm25l_train.pkl
│   └── aclarc_bm25l_train.pkl
└── processed/
    ├── scicite/                # SciCite splits: train/dev/test (+ label2 variants)
    ├── aclarc/                 # ACL-ARC cross-domain splits
    └── augmented_datasets.zip  # cached retrieval-augmented inputs
                                # (unzip in place to obtain cache/augment_*.pkl)
```

## Regenerating

```
python scripts/download_data.py           # fetch the raw SciCite / ACL-ARC corpora
python scripts/preprocess.py              # rebuild data/processed/
python scripts/build_retrieval_index.py   # rebuild data/index/
```

`data/raw/` and `data/processed/cache/` are intentionally not tracked: the
former is re-downloadable, the latter is shipped as `augmented_datasets.zip`.