#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=4
#PJM -L elapse=01:00:00
#PJM -j

module load cuda cudnn nccl gcc

# GPU情報を表示
nvidia-smi
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
echo "Number of GPUs: $GPU_COUNT"

uv sync

# マルチGPUでエンコーディング実行
uv run python src/preprocess/triplet-ja_encode.py \
    --data_name "cl-nagoya/ruri-dataset-v2-pt" \
    --teacher_model "Qwen/Qwen3-Embedding-4B" \
    --long_batch_size 2 \
    --short_batch_size 64 \
    --max_length 4096 \
    --sample_size 1_000_000