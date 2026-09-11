# SciCite-REAL

**When Retrieval Does Not Help: A Controlled Study of Citation Intent
Classification with BM25L Evidence.**

Code, configurations and run logs for a leakage-controlled study of BM25L
retrieval augmentation (CiteREAL) over three pretrained encoders on SciCite.
Everything trains and evaluates on **6 GB of VRAM (RTX 3060)**.

Headline (SciCite test split, seed 42):

| model | acc | macro-F1 |
|---|---|---|
| BERT-base | 85.71 | 84.11 |
| SciBERT | 86.30 | 85.11 |
| **DeBERTa-v3-base** | **86.89** | **85.49** |
| DeBERTa + retrieval, CLS head | 59.97 | 38.73 |
| DeBERTa + retrieval, dual head | 84.95 / 86.14 | 83.30 / 84.71 |
| DeBERTa + retrieval, dual head, leak-free mask | 96.40 | 95.04 (elimination leak, see 3) |

No augmented variant beats its plain encoder under the leakage controls below.
Three side findings matter if you run this recipe yourself:

1. **A bare `[CLS]` head collapses on augmented inputs** (59.97% accuracy).
   The gated dual-pooling head (`--pooling dual`) restores the baseline.
2. **Mixed fp16/fp32 weights fail silently.** The backbone must load in fp32
   with contiguous parameters; otherwise the run drifts to majority-class
   prediction while the loss keeps decreasing.
3. **The leak-free mask is itself a leak.** Forbidding same-label evidence
   pushes accuracy to 96.40% — with 3 classes and k=3, the target label is
   exactly the class absent from its neighbours, and the model learns the
   elimination rule (train acc 97.88%). Any mask conditioned on the label
   changes the input in label-correlated ways; our default protocol
   retrieves label-blind.

---

## Highlights
- **Leakage-safe retrieval**: train-only pool, O(1) self-exclusion, optional
  `--leak-free` mask (forbids same-label evidence — itself an elimination
  leak, see finding 3).
- **Gated dual-pooling head** fusing `[CLS]` with mean-pooled evidence.
- **Vectorised BM25L** (`src/retrieval/fast_bm25l.py`): scores agree with
  `rank_bm25.BM25L` to float64 rounding, evidence byte-identical; the full
  augmented build over 8,243 contexts runs in ~12 s instead of ~8 min.
- **Disk-cached augmentation**: reruns and ablations load byte-identical
  evidence in under a second.
- 6 GB recipe: bf16 + gradient accumulation + gradient checkpointing; fp32
  backbone load (see finding 2 above).
- Metrics: accuracy, macro-F1, per-class report, confusion matrix.

## Installation
```bash
git clone https://github.com/Merthyr04/SciCite-REAL.git && cd SciCite-REAL
python -m venv .venv && source .venv/bin/activate   # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

## Quick start (5 minutes, CPU / small GPU)
```bash
python scripts/download_data.py                 # fetch SciCite tarball
python scripts/preprocess.py                    # -> data/processed/scicite
python scripts/bench_fast_bm25l.py              # verify + time the fast scorer
python notebooks/demo.ipynb                     # watch BM25 evidence materialise
```

## Reproduce the full paper (GPU)
```bash
bash scripts/run_all.sh                          # baselines + REAL + ablations + eval
python scripts/collect_metrics.py runs           # -> markdown results table
```
Manual step-by-step:
```bash
# 0. data + index
python scripts/download_data.py
python scripts/preprocess.py --raw data/raw/scicite --out data/processed/scicite
python scripts/build_retrieval_index.py --data data/processed/scicite --split train \
       --out data/index/scicite_bm25l_train.pkl

# 1. baselines
python scripts/run_training.py --model bert   --tag baseline_bert
python scripts/run_training.py --model scibert --tag baseline_scibert
python scripts/run_training.py --model deberta --tag baseline_deberta

# 2. retrieval (note: use --pooling dual; the CLS head is the known failure mode)
python scripts/run_training.py --model deberta --retrieval --index data/index/scicite_bm25l_train.pkl \
       --top-k 3 --pooling dual --tag real_deberta_dual_k3

# 3. ablations
python scripts/run_training.py --model deberta --retrieval --max-length 128 --top-k 1 --pooling dual --tag ablation_len128_k1
python scripts/run_training.py --model deberta --retrieval --leak-free  --pooling dual --tag ablation_deberta_leakfree
```

## Retrieval preprocessing: reference vs ours
`scripts/bench_fast_bm25l.py` scores all 8,243 training contexts against the
pool with both `rank_bm25.BM25L` and `src/retrieval/fast_bm25l.py`, compares
scores and top-3 rankings, and writes `runs/fast_bm25l_bench.json`:

```bash
python scripts/bench_fast_bm25l.py
```

| metric | rank_bm25 (reference) | FastBM25L (ours) |
|---|---|---|
| full scoring pass, 8,243 queries | 420.8 s | 5.5 s |
| full augmented build (3 splits) | 498.2 s | 11.6 s |
| max abs score diff | — | 5.5e-12 |
| top-3 ranking mismatches / evidence diffs | — | 0 / 0 |
| cached augmented rebuild | — | 0.5 s |

Measured training throughput (DeBERTa-v3, exact recipe above): plain 34.1 vs.
augmented 34.4 samples/s — the augmentation is free at training time, because
the plain baseline pads every context to 256 tokens although the median
context is 48 tokens long, so both configurations process ~2.1M tokens per
epoch. See `runs/train_throughput_bench.json`.

## Project layout
```
SciCite-REAL/
├── src/
│   ├── data/          # loading, preprocessing
│   ├── retrieval/     # BM25 index, FastBM25L, RetrievalAugmenter
│   ├── models/        # encoder + classification head
│   ├── training/      # Trainer wrapper + utils
│   └── evaluation/    # metrics, confusion matrix, reports
├── scripts/           # CLI entry points (download/preprocess/index/train/eval/bench)
├── configs/           # YAML for each table row
├── notebooks/         # end-to-end demo
├── docs/              # DATA_notes.md, METHODOLOGY.md
├── paper/             # LaTeX (CVPR-style) + HTML preprint
└── data/              # downloaded & processed (gitignored)
```

## Memory tuning on a 6 GB card
Prefer the defaults (`--batch-size 8 --grad-accumulation 4`, bf16, gradient
checkpointing). If OOM persists: `--max-length 128`, `--freeze-embeddings`,
or a smaller backbone. Do **not** trade the fp32 backbone load for speed —
see finding 2 above.

## Results
Run `python scripts/collect_metrics.py runs` to regenerate from your own
machine. The numbers in the table at the top come from `runs/*/metrics.json`
in this repository.

## Data & license
- **SciCite** (Cohan et al., NAACL 2019) — Apache-2.0. See `docs/DATA_notes.md`.
- **ACL-ARC** (Jurgens et al., 2018) — optional cross-dataset.
- Code: Apache-2.0. See `LICENSE`.

## Cite
```bibtex
@article{boxin2026citereal,
  title  = {When Retrieval Does Not Help: A Controlled Study of Citation
            Intent Classification with BM25L Evidence},
  author = {Boxin, Chen},
  year   = {2026},
  note   = {Preprint}
}
```
If you enrich inputs with cited-paper abstracts via `api.semanticscholar.org`,
include Semantic Scholar API attribution.
