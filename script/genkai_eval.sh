#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=3:00:00
#PJM -j
#PJM -o logs/genkai_eval.log

module load cuda cudnn nccl gcc

nvidia-smi
uv sync
for model_name in "nomic-ai/modernbert-embed-base-unsupervised"; do
    uv run python eval_mteb.py \
        --model_name $model_name \
        --batch_size 128 \
        --benchmark_name "MTEB(eng, v2)"
done