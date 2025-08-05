#!/bin/bash

python train.py \
    --student_model cl-nagoya/ruri-v3-pt-30m \
    --teacher_model Qwen/Qwen3-Embedding-4B \
    --data_path data/train \
    --output_dir output \
    --batch_size 8 \
    --num_epochs 5 \
    --loss_type taid \
    --taid_t_start 0.2 \
    --taid_alpha 5.e-4 \
    --taid_beta 0.99  