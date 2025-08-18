#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=4
#PJM -L elapse=10:00:00
#PJM -j

module load cuda cudnn nccl gcc

# GPU情報を表示
nvidia-smi
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
echo "Number of GPUs: $GPU_COUNT"

uv sync

# マルチGPUでエンコーディング実行
uv run python src/preprocess/triplet-en_encode.py \
    --teacher_model "Qwen/Qwen3-Embedding-0.6B" \
    --long_batch_size 8 \
    --short_batch_size 128 \
    --max_length 4096 \
    --sample_size 1_000_000