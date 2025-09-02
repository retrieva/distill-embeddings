#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=3:00:00
#PJM -j
#PJM -o logs/0902/02.log

module load cuda cudnn nccl gcc

nvidia-smi
uv sync
for model_name in "nomic-ai/modernbert-embed-base-unsupervised"; do
    uv run python src/evaluation/run_mteb.py \
        --model_name 'output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554/gte_e3_bs128_cosine5e-05_kld_w-pos/checkpoints/epoch=02.ckpt' \
        --batch_size 128 \
        --benchmark_name "MTEB(eng, v2)"
