#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=07:00:00
#PJM -j

module load cuda cudnn nccl gcc

# GPU情報を表示
nvidia-smi

uv sync

uv run python src/preprocess/triplet-en_encode.py \
    --teacher_model "Qwen/Qwen3-Embedding-0.6B" \
    --long_batch_size 2 \
    --short_batch_size 64 \
    --max_length 4096 \
    --sample_size 3_000_000 \
    --disable_multigpu