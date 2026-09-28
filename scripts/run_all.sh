#!/usr/bin/env bash
# One-shot reproduction of every table in the paper on an 8GB+ GPU.
# On a 6 GB RTX 3060 keep batch_size=8 and grad_accumulation=4 (fp16 handles the rest).
#
# Usage:  bash scripts/run_all.sh
set -euo pipefail

DATA=data/processed/scicite
INDEX=data/index/scicite_bm25l_train.pkl

echo "== 0. Pre-flight =="
python scripts/download_data.py --data-dir data/raw
python scripts/preprocess.py --raw data/raw/scicite --out "$DATA"
python scripts/build_retrieval_index.py --data "$DATA" --split train --out "$INDEX"

echo
echo "== 1. Baselines (no retrieval) =="
python scripts/run_training.py --data "$DATA" --model bert   --tag baseline_bert   --dropout 0.1
python scripts/run_training.py --data "$DATA" --model scibert --tag baseline_scibert --dropout 0.1
python scripts/run_training.py --data "$DATA" --model deberta --tag baseline_deberta --dropout 0.1

echo
echo "== 2. Proposed: DeBERTa + BM25 (top_k=3) =="
python scripts/run_training.py --data "$DATA" --model deberta --retrieval \
    --index "$INDEX" --top-k 3 --dropout 0.2 --tag real_deberta_k3

echo
echo "== 2b. Structural upgrade: DeBERTa + BM25 + dual-pooling head =="
python scripts/run_training.py --data "$DATA" --model deberta --retrieval \
    --index "$INDEX" --top-k 3 --dropout 0.2 --pooling dual --tag real_deberta_dual_k3

echo
echo "== 3. Ablations =="
# 3a. retrieval off is baseline_deberta above; here the sequence-length & top_k study:
python scripts/run_training.py --data "$DATA" --model deberta --retrieval \
    --index "$INDEX" --top-k 1 --max-length 128 --dropout 0.2 --tag ablation_len128_k1
# 3b. leak-free retrieval:
python scripts/run_training.py --data "$DATA" --model deberta --retrieval \
    --index "$INDEX" --top-k 3 --leak-free --dropout 0.2 --tag ablation_deberta_leakfree

echo
echo "== 4. Evaluate checkpoints =="
python scripts/run_evaluation.py --checkpoint runs/baseline_deberta/best_model --data "$DATA" --tag baseline_deberta --report
python scripts/run_evaluation.py --checkpoint runs/real_deberta_k3/best_model --data "$DATA" --tag real_deberta_k3 --report

echo
echo "All runs finished. Collect metrics with:  python scripts/collect_metrics.py runs"