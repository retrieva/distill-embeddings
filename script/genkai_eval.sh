#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -j

module load cuda cudnn nccl gcc

nvidia-smi
uv sync
for model_name in "answerdotai/ModernBERT-base" "answerdotai/ModernBERT-large"; do
    uv run python eval_mteb.py \
        --model_name $model_name \
        --batch_size 64
done