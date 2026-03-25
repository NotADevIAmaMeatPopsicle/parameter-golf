#!/bin/bash
# v8.1 Scoring Run — NO TTT (legal, fast, submittable)
# Total time: ~14 min (10 train + 2 compress + 2 eval)
set -e

export DATA_PATH=data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=data/tokenizers/fineweb_1024_bpe.model
export MAX_WALLCLOCK_SECONDS=600
export USE_COMPILE=1
export SEED=${1:-1337}

# v8.1 optimal config
export NUM_LAYERS=12
export MLP_HIDDEN=1300
export BIGRAM_VOCAB_SIZE=8192
export MATRIX_LR=0.02
export XSA_LAYERS=12
export GATED_ATTENTION=0
export VALUE_RESIDUAL=1
export STP_LAMBDA=0.02
export CROWN_Q_LAMBDA=0.01

# Compression: codebook only (GPTQ hurts)
export USE_NOVEL_COMPRESSION=1
export USE_GPTQ=0

# NO TTT (keeps eval under 600s)
export TTT_ENABLED=0

# Eval
export EVAL_STRIDE=64
export NGRAM_EVAL=0

# Off
export PROGRESSIVE_SEQ=0
export USE_TRIGRAM=0
export USE_QUADGRAM=0
export SPARSE_HIDDEN_DIM=0

echo "=== v8.1 SCORING RUN seed=$SEED $(date) ==="
python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-23_pcloadloveletter_v7/train_gpt.py \
  2>&1 | tee /workspace/run_seed${SEED}.log
echo "=== DONE $(date) ==="
