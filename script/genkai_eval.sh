#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=3:00:00
#PJM -j

module load cuda cudnn nccl gcc

nvidia-smi
uv sync
for model_name in "distill-embeddings/output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554/gte_e3_bs128_wsd5e-05_taid-kld_w-pos/checkpoints/epoch=02.ckpt"; do
    uv run python eval_mteb.py \
        --model_name $model_name \
        --batch_size 128
done