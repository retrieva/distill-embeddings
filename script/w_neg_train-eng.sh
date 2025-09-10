#!/bin/bash

# read -p "Enter loss_type: " loss_type
#     echo "Running with loss_type: $loss_type"
# 
    
    # Run the training script with the specified loss type
for loss_type in "kld"; do
# for loss_type in "mse" "kld" "taid-mse" "taid-kld"; do
# for loss_type in "mse" "kld" "ckd" "taid-ckd" "taid-mse" "taid-kld"; do
    uv run python -m src.training.train \
        --student_model nomic-ai/modernbert-embed-base-unsupervised \
        --teacher_model Qwen/Qwen3-Embedding-4B \
        --data_size 1794550 \
        --data_name gte_en_plus_w_neg \
        --batch_size 4 \
        --num_epochs 1 \
        --max_length 32 \
        --language eng \
        --use_pos \
        --use_neg \
        --mteb_eval \
        --add_prefix True \
        --loss_type "$loss_type" 
done
        # --get_id_iso \