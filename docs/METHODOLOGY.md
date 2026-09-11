# Methodology — retrieval-augmented citation-intent classification

## Problem
Given the **citing context** (the sentence around a citation marker) we predict
the intent of the citation: *background, method, or result*.

## Baseline model
```
text ──► Transformer encoder (BERT / SciBERT / DeBERTa*) ──► [CLS] ──► MLP ──► logits(3)
```
- shared backbone + linear classification head across all models (fair comparison)
- Sequence length 256 tokens

## Retrieval augmentation (CiteREAL)
A citation intent is often recoverable from *how* the context is phrased, and
similar phrases recur across papers with the same intent. We retrieve the
**top-k most similar citation contexts** from a retrieval pool (default: the
training corpus) with **BM25L** (Lv & Zhai, 2011) and prepend them as evidence:

```
[evidence_1] [evidence_2] [evidence_3]  [TARGET citing context]
                    ─────────  ────────────────────────────────
                    retrieved (k=3)          original input
```

## Gated dual-pooling head (`--pooling dual`)
Concatenated evidence breaks a bare `[CLS]` head (59.97% accuracy in our runs).
The dual head computes two views — `[CLS]` for the target sentence and
mean-pooling over evidence token positions — then fuses them with a learned
scalar gate before the linear head. One extra linear layer of parameters.

## Retrieval details (BM25L)
- Tokenisation: lowercase, strip punctuation, fold whitespace
- `idf(w) = log(N+1) - log(df(w)+0.5)`,
  `ctd = tf / (1 - b + b·|d|/avgdl)`,
  `score = Σ_w idf·tf·(k1+1)·(ctd+δ) / (k1+ctd+δ)`
- `k1 = 1.5, b = 0.75, delta = 0.5`

### Fast scorer
`src/retrieval/fast_bm25l.py` evaluates the same arithmetic over a sparse CSC
term-frequency matrix (column slices touch only nonzeros). Over all 8,243
training queries the scores agree with `rank_bm25.BM25L` to float64 rounding
(max |Δ| = 5.5e-12), every query returns the same top-3, and the retrieved
evidence is byte-identical. Measured on the RTX 3060 machine: scoring pass
420.8 s → 5.5 s (76×), full augmented build 498.2 s → 11.6 s (43×), cached
rebuild 0.5 s. `scripts/bench_fast_bm25l.py` verifies equivalence and writes
`runs/fast_bm25l_bench.json`; `scripts/bench_train_throughput.py` measures
training throughput (plain 34.1 vs. dual 34.4 samples/s on the same token
budget) and writes `runs/train_throughput_bench.json`.

## Leakage control (important!)
- The **retrieval pool is built from the TRAIN split only** → test instances are
  never retrieved-from, so the evidence never contains the true label verbatim.
- We exclude the target context itself (`exclude_self`, O(1) via text→index map).
- **Leak-free ablation**: optionally forbid retrieving contexts whose label
  equals the target's, isolating the label-copying failure path. Measured
  result: the mask *itself* leaks — with 3 classes and k=3, the surviving
  neighbours carry only the other two classes, so the target label is the
  class absent from the input; the model learns this elimination rule and
  reaches 96.40% test / 97.88% train accuracy. Never treat the leak-free
  number as a performance gain; the default protocol retrieves label-blind.

## 6 GB VRAM training recipe (RTX 3060)
| knob | value | why |
|------|-------|-----|
| `bf16` mixed precision | on | avoids fp16 GradScaler + checkpointing issues |
| `--batch-size 8` + `--grad-accumulation 4` | effective batch 32 | gradient quality with small memory |
| `--max-length 256` | short citation snippets | memory + speed |
| `gradient_checkpointing` | on | saves ~40% activation RAM |
| **fp32 backbone load** | required | mixed fp16/fp32 weights collapse to majority-class silently (53.57% acc / 23.26 macro-F1 in our runs); load with `torch_dtype=torch.float32` and `.contiguous()` |
| `freeze_embeddings` | optional | frees embedding matrix |
| `deberta-small` | alternative | even lighter backbone |

## Disk cache for augmented data
`scripts/run_training.py` pickles the augmented dataset once per retrieval
configuration (key: `sha1(top_k, leak_free, index path)`). Reruns and ablations
reusing the configuration load byte-identical evidence in under a second.

## Evaluation
- **Accuracy** (aligns with original SciCite)
- **Macro-F1** (robust to `background`-heavy imbalance)
- ablations: head (CLS vs dual) / dtype stability / `max_length` / `top_k` / leak-free
