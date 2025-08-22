#!/bin/sh
#PJM --interact
#PJM -L rscgrp=b-inter
#PJM -L gpu=1
#PJM -L elapse=1:00:00
module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

uv sync

uv run python get_isoscore.py \
    --model_name "Qwen/Qwen3-Embedding-4B" \
    --prompt_name "document"

uv run python get_isoscore.py \
    --model_name "Qwen/Qwen3-Embedding-4B" \
    --prompt_name "query"

uv run python get_isoscore.py \
    --model_name "Qwen/Qwen3-Embedding-8B" \
    --prompt_name "document"

uv run python get_isoscore.py \
    --model_name "Qwen/Qwen3-Embedding-8B" \
    --prompt_name "query"