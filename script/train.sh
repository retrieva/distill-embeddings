#!/bin/bash

uv run python train.py \
    --student_model cl-nagoya/ruri-v3-pt-30m \
    --teacher_model Qwen/Qwen3-Embedding-4B \
    --data_dir data \
    --dataset_name sample_10BT_100000 \
    --output_dir output \
    --batch_size 64 \
    --num_epochs 1 \
    --max_length 4096 \
    --val_check_interval 0.1 \
    --validate_first \
    --loss_type ckd