#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=72:00:00
#PJM -j
#PJM -o logs/0819/07.log

module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

for loss_type in "ckd"; do
    uv run python train.py \
        --student_model answerdotai/ModernBERT-base \
        --teacher_model Qwen/Qwen3-Embedding-0.6B \
        --data_dir data \
        --dataset_name 1000000 \
        --output_dir output/result \
        --batch_size 64 \
        --num_epochs 2 \
        --max_length 512 \
        --val_check_interval 0.1 \
        --log_every_n_steps 1 \
        --language "eng" \
        --taid_t_start 0.6 \
        --mteb_eval \
        --use_pos \
        --loss_type "$loss_type"
done