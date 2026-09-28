# SciCite-REAL

**Citation Intent Classification under Retrieval Augmentation: A Controlled
Study and Its Implications for Scientometric Applications.**

Code, configurations, run logs and derived artifacts for a leakage-controlled
study of retrieval augmentation (BM25 / BM25L evidence) over three pretrained
encoders on SciCite, with a cross-domain replication on ACL-ARC. Everything
trains and evaluates on **6 GB of VRAM (RTX 3060)**.

Paper under review at *Scientometrics* (Springer Nature).

## Headline results

SciCite test split (1,861 instances), seed 42. Two runs are reported for
DeBERTa and for the augmented dual-head model, because the run-to-run spread is
the resolution at which this comparison has to be read.

| model | acc | macro-F1 |
|---|---|---|
| BERT-base | 85.71 | 84.11 |
| SciBERT | 86.30 | 85.11 |
| DeBERTa-v3-base (run 1) | **86.89** | **85.49** |
| DeBERTa-v3-base (run 2) | 85.98 | 84.22 |
| DeBERTa + BM25L, `[CLS]` head (right-truncated) | 59.97 | 38.73 |
| DeBERTa + BM25L, dual head (run 1) | 85.60 | 84.20 |
| DeBERTa + BM25L, dual head (run 2) | 86.03 | 84.58 |
| BERT + BM25L, dual head | 84.85 | 83.22 |
| SciBERT + BM25L, dual head | 86.24 | 85.01 |

**No augmented variant beats its own plain encoder.** Neither augmented
DeBERTa run reaches the backbone's better run (86.89 / 85.49), and the
augmented pair straddles the backbone's weaker run (85.98 / 84.22). The same
holds for the other two backbones, which move from 85.71 to 84.85 (BERT) and
from 86.30 to 86.24 (SciBERT).

## What the study finds

Four results, all of which matter if you run this recipe yourself.

1. **The `[CLS]` collapse is a truncation bug, not a head limitation.** The
   naive concatenation places retrieved evidence in front of a short target and
   truncates from the right. Because the BM25L neighbours average 1,813
   characters, that deletes the target sentence — the only part of the input
   carrying the label — in **95.9% of training rows and 98.5% of test rows**,
   and the model lands at 59.97% while the loss curve stays healthy. Holding the
   head, evidence and seed fixed and changing *only* the truncation policy
   recovers the model to **85.87 / 84.51**. The dose-response is visible in the
   other direction too: BM25 returns neighbours averaging 236 characters, loses
   the target in only 5.4% of rows, and its right-truncated `[CLS]` run scores
   83.07%. `scripts/_diag_cls_truncation.py` is the diagnostic that counts
   surviving targets; it is the cheapest possible check for this hazard.

2. **The leak-free mask is itself a leak.** Forbidding same-label evidence
   pushes accuracy to **96.02%** — with 3 classes and k=3 the target label is
   exactly the class *absent* from the neighbours, so the model learns the
   elimination rule rather than the task (train acc 98.05%). Any input
   construction conditioned on the label changes the input distribution in
   label-correlated ways, whether it adds evidence or removes it. Our default
   protocol retrieves label-blind.

3. **Intent is strongly conditioned on rhetorical section.** A section prior
   alone predicts **73.99%** of the SciCite test labels, against 53.57% for the
   global majority class. Method citations cluster in Methods (80.7% of that
   section's contexts), result citations in Discussion (40.2%), background
   citations in Introduction and Related work (88.2% and 84.5%). This is why a
   strong encoder already recovers nearly all of the remaining signal from the
   target sentence, and why retrieval has little left to add.

4. **Swapping the retriever does not change the conclusion.** BM25 and BM25L
   applied to the same pool with the same self-exclusion control return
   different evidence — their top-3 sets coincide for only 0.9% of training
   queries — yet every between-scorer gap is no larger than the spread that
   repeat runs of a single scorer produce. See the BM25 comparison table in the
   paper and `runs/paper_numbers.json`.

## Ablations

SciCite, DeBERTa-v3 with BM25L and k=3, seed 42. Same data, one variable at a
time.

| configuration | acc | macro-F1 |
|---|---|---|
| `[CLS]` head, right-truncated | 59.97 | 38.73 |
| `[CLS]` head, target-preserving | 85.87 | 84.51 |
| dual head (ours) | 85.60 | 84.20 |
| leak-free mask (dual head) | 96.02 | 94.48 |

The leak-free row is an elimination leak, not a gain, and is reported to
document the failure mode. The honest comparison is the label-blind rows.

## Cross-domain replication (ACL-ARC)

Six frames, 139 test instances. Absolute scores carry wide intervals at this
size and are read qualitatively.

| model | acc | macro-F1 |
|---|---|---|
| DeBERTa-v3-base, no retrieval | 67.63 | 31.36 |
| + BM25L, `[CLS]` head | 51.08 | 12.44 |
| + BM25L, dual head | 56.12 | 21.98 |
| + BM25L, dual head, leak-free | 61.15 | 27.73 |

The negative result replicates. The elimination leak does not: with six classes
three neighbours can exclude at most half the label space, so the shortcut is
available only in proportion to how much of the label space the evidence can
exclude.

## Highlights

- **Leakage-safe retrieval**: train-only pool, O(1) self-exclusion, optional
  `--leak-free` mask (forbids same-label evidence — itself an elimination leak,
  see finding 2).
- **Gated dual-pooling head** fusing `[CLS]` with mean-pooled evidence.
- **`--preserve-target` truncation**: drops evidence tokens instead of the
  target sentence. This is the fix for finding 1.
- **Vectorised BM25L** (`src/retrieval/fast_bm25l.py`): scores agree with
  `rank_bm25.BM25L` to float64 rounding and evidence is byte-identical; the full
  augmented build over 8,243 contexts runs in ~12 s instead of ~8 min.
- **Disk-cached augmentation**: reruns and ablations load byte-identical
  evidence in under a second.
- 6 GB recipe: bf16 + gradient accumulation + gradient checkpointing, fp32
  backbone load.
- Metrics: accuracy, macro-F1, per-class report, confusion matrix, and error
  analysis by section and by context length.

## Installation

```bash
git clone https://github.com/Merthyr04/SciCite-REAL.git && cd SciCite-REAL
python -m venv .venv && source .venv/bin/activate   # (or .venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

Tested on Python 3.10, Windows and Ubuntu, CUDA 11.8. A 6 GB GPU is enough;
the quick start below runs on CPU.

## Quick start (5 minutes, CPU / small GPU)

```bash
python scripts/download_data.py                 # fetch SciCite tarball
python scripts/preprocess.py                    # -> data/processed/scicite
python scripts/bench_fast_bm25l.py              # verify + time the fast scorer
python notebooks/demo.ipynb                     # watch BM25 evidence materialise
```

## Reproduce the paper

Every number in the tables above comes from `runs/*/metrics.json` in this
repository. The full pipeline:

```bash
# 0. data + index
python scripts/download_data.py
python scripts/preprocess.py --raw data/raw/scicite --out data/processed/scicite
python scripts/build_retrieval_index.py --data data/processed/scicite --split train \
       --out data/index/scicite_bm25l_train.pkl --variant bm25l
python scripts/build_retrieval_index.py --data data/processed/scicite --split train \
       --out data/index/scicite_bm25_train.pkl --variant bm25

# 1. baselines
python scripts/run_training.py --model bert    --tag baseline_bert
python scripts/run_training.py --model scibert --tag baseline_scibert
python scripts/run_training.py --model deberta --tag baseline_deberta

# 2. augmented runs — use --pooling dual; the [CLS] head is the failure mode
python scripts/run_training.py --model deberta --retrieval \
       --index data/index/scicite_bm25l_train.pkl --top-k 3 --pooling dual \
       --tag real_deberta_dual_k3
python scripts/run_training.py --model bert    --retrieval \
       --index data/index/scicite_bm25l_train.pkl --top-k 3 --pooling dual \
       --tag bert_ret_dual
python scripts/run_training.py --model scibert --retrieval \
       --index data/index/scicite_bm25l_train.pkl --top-k 3 --pooling dual \
       --tag scibert_ret_dual

# 3. the truncation control (finding 1): same head, same evidence, target kept
python scripts/run_training.py --model deberta --retrieval \
       --index data/index/scicite_bm25l_train.pkl --top-k 3 --pooling cls \
       --preserve-target --tag real_deberta_k3_pt

# 4. the leak-free control (finding 2)
python scripts/run_training.py --model deberta --retrieval \
       --index data/index/scicite_bm25l_train.pkl --top-k 3 --pooling dual \
       --leak-free --tag ablation_real_deberta_leakfree

# 5. collation and analysis
python scripts/paper_numbers.py --out runs/paper_numbers.json
python scripts/collect_metrics.py runs --out runs/results_summary.md
python scripts/error_analysis.py --checkpoint runs/baseline_deberta/best_model \
       --tag err_baseline_deberta
python scripts/distribution_analysis.py
```

On Windows the serial runners `run_dual_rerun.ps1` and `run_cls_control.ps1`
execute the augmented grid and the `[CLS]` controls back to back without GPU
contention.

### Diagnostic for finding 1

```bash
python scripts/_diag_cls_truncation.py
```

Counts, for both truncation policies, how many training and test targets are
fully or partially deleted by the context window. This is the measurement that
separates "the head is fragile" from "the target was never in the input".

### Reference implementation of the scorer

`scripts/bench_fast_bm25l.py` scores all 8,243 training contexts against the
pool with both `rank_bm25.BM25L` and `src/retrieval/fast_bm25l.py`, compares
scores and top-3 rankings, and writes `runs/fast_bm25l_bench.json`:

| metric | rank_bm25 (reference) | FastBM25L (ours) |
|---|---|---|
| full scoring pass, 8,243 queries | 420.8 s | 5.5 s |
| full augmented build (3 splits) | 498.2 s | 11.6 s |
| max abs score diff | — | 5.5e-12 |
| top-3 ranking mismatches / evidence diffs | — | 0 / 0 |
| cached augmented rebuild | — | 0.5 s |

Measured training throughput (DeBERTa-v3, exact recipe above): plain 34.1 vs.
augmented 34.4 samples/s, i.e. an overhead of −0.9%. The augmentation is free
at training time, because the plain baseline pads every context to 256 tokens
although the median context is about 30 words long, so both configurations
process roughly the same number of tokens per epoch. See
`runs/train_throughput_bench.json`.

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
├── configs/           # example YAML configs (exact flags used: see Reproduce the paper)
├── notebooks/         # end-to-end demo
├── docs/              # DATA_notes.md, METHODOLOGY.md
├── paper/             # Scientometrics submission (Springer sn-jnl) + earlier venue drafts
├── runs/              # per-run metrics.json and analysis JSONs
└── data/              # downloaded & processed (gitignored; see Data below)
```

## Memory tuning on a 6 GB card

Prefer the defaults (`--batch-size 8 --grad-accumulation 4`, bf16, gradient
checkpointing). If OOM persists: `--max-length 128`, `--freeze-embeddings`, or a
smaller backbone.

## Data & license

- **SciCite** (Cohan et al., NAACL 2019) — Apache-2.0. See `docs/DATA_notes.md`.
- **ACL-ARC** (Jurgens et al., 2018) — cross-domain benchmark, Apache-2.0.
- Code: Apache-2.0. See `LICENSE`.
- Derived splits, retrieval indices, the raw `metrics.json` behind every table,
  and the analysis JSONs in `runs/` are released with this repository.

If you enrich inputs with cited-paper abstracts via `api.semanticscholar.org`,
include Semantic Scholar API attribution.

## Cite

```bibtex
@article{chen2026citationintent,
  title  = {Citation Intent Classification under Retrieval Augmentation:
            A Controlled Study and Its Implications for Scientometric
            Applications},
  author = {Chen, Boxin},
  year   = {2026},
  note   = {Manuscript under review at Scientometrics}
}
```