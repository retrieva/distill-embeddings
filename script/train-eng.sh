#!/bin/bash

# read -p "Enter loss_type: " loss_type
#     echo "Running with loss_type: $loss_type"
# 
    
    # Run the training script with the specified loss type
for loss_type in "kld"; do
# for loss_type in "mse" "kld" "taid-mse" "taid-kld"; do
# for loss_type in "mse" "kld" "ckd" "taid-ckd" "taid-mse" "taid-kld"; do
    uv run python train.py \
        --student_model answerdotai/ModernBERT-base \
        --teacher_model Qwen/Qwen3-Embedding-4B \
        --data_dir data \
        --dataset_name 2000 \
        --output_dir output/result \
        --batch_size 4 \
        --num_epochs 1 \
        --max_length 4096 \
        --val_check_interval 1 \
        --log_every_n_steps 1 \
        --language eng \
        --get_id_iso \
        --use_pos \
        --loss_type "$loss_type"
done
        # --mteb_eval \