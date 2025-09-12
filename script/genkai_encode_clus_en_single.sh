#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=20:00:00
module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")


CUDA_VISIBLE_DEVICES=7 uv run python -m src.data_processing.w_label_data.encode_mixed \
    --base-dir data/w_label_data/mixed_triplets_hf \
    --output-dir data/w_label_data/Qwen_Qwen3-Embedding-4B_encoded \
    --teacher-model Qwen/Qwen3-Embedding-4B \
    --threshold 2048 \
    --long-batch-size 1 \
    --short-batch-size 32 \
    --max-length 4096 \
    --num-proc 4