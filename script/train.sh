#!/bin/bash

uv run python train.py \
    --student_model cl-nagoya/ruri-v3-pt-30m \
    --teacher_model Qwen/Qwen3-Embedding-4B \
    --data_dir data \
    --dataset_name sample_10BT_1000 \
    --output_dir output \
    --batch_size 16 \
    --num_epochs 3 \
    --max_length 4096 \
    --val_check_interval 1 \
    --log_every_n_steps 1 \
    --loss_type ckd
    # --taid_forward_fn ckd \
    # --mteb_eval \
