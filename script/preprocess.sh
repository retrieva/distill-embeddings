#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=07:00:00
#PJM -j

module load cuda cudnn nccl gcc

# GPU情報を表示
nvidia-smi

uv sync
uv run python -m src.data_processing.finweb.encode \
    --data_name "hotchpotch/fineweb-2-edu-japanese" \
    --subset_name "sample_10BT" \
    --sample_size 1794545 \
    --output_dir "data" \
    --teacher_model "Qwen/Qwen3-Embedding-4B" \
    --long_batch_size 2 \
    --short_batch_size 64 \
    --threshold 2048 \
    --max_length 4096
