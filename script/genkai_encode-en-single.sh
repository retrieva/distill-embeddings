#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=07:00:00
#PJM -j

module load cuda cudnn nccl gcc

# GPU情報を表示
nvidia-smi

uv sync

uv run python -m src.preprdata_processingocess.gte_en_plus_data.encode \
    --teacher_model "Qwen/Qwen3-Embedding-4B" \
    --long_batch_size 2 \
    --short_batch_size 64 \
    --max_length 4096 \
    --sample_size 1_000_000 \
    --w_instruction \
    --disable_multigpu