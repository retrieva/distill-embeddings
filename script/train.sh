#!/bin/bash

# read -p "Enter loss_type: " loss_type
#     echo "Running with loss_type: $loss_type"
    
    # Run the training script with the specified loss type
for loss_type in "mse" "kld" "taid-mse" "taid-kld"; do
# for loss_type in "mse" "kld" "ckd" "taid-ckd" "taid-mse" "taid-kld"; do
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
        --loss_type "$loss_type"
        # --mteb_eval \
done