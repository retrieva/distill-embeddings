#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -j
#PJM -o logs/0903/07.log

module load cuda cudnn nccl gcc

nvidia-smi
uv sync
model_paths=(
"output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/794554/gte_e10_bs128_wsd0.0001_taid-kld_w-pos_prefix/checkpoints/epoch=09.ckpt"
)
for model_path in ${model_paths[@]}; do
    uv run python -m src.evaluation.run_mteb \
        --model_name $model_path \
        --batch_size 128 \
        --benchmark_name "on_train_end_tasks"
done