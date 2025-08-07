#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=01:00:00
#PJM -j

module load cuda cudnn nccl gcc

source venv/bin/activate  # 前段で構築した環境を読み込む

uv run python train.py \
    --student_model cl-nagoya/ruri-v3-pt-30m \
    --teacher_model Qwen/Qwen3-Embedding-4B \
    --data_dir data \
    --dataset_name sample_10BT_100000 \
    --output_dir output \
    --batch_size 64 \
    --num_epochs 3 \
    --max_length 4096 \
    --val_check_interval 0.1 \
    --log_every_n_steps 1 \
    --mteb_eval \
    --loss_type "$loss_type"