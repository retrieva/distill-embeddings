#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=10:00:00
#PJM -j
#PJM -o logs/0826/01.log

module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

for loss_type in "taid-kld","kld"; do
    uv run python train.py \
        --student_model answerdotai/ModernBERT-base \
        --teacher_model Qwen/Qwen3-Embedding-4B \
        --data_dir data \
        --dataset_name 50000 \
        --output_dir output/result \
        --batch_size 128 \
        --num_epochs 3 \
        --max_length 512 \
        --val_check_interval 0.1 \
        --log_every_n_steps 1 \
        --mteb_eval \
        --language "eng" \
        --taid_t_start 0.6 \
        --scheduler "wsd" \
        --use_pos \
        --loss_type "$loss_type"
done