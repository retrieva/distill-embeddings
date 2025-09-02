#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=20:00:00
#PJM -j
#PJM -o logs/0902/11.log

module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

for loss_type in "taid-kld"; do
    for lr in 1e-4; do
        for distill_weight in 0.7;do
            uv run python -m src.training.train \
                --student_model nomic-ai/modernbert-embed-base-unsupervised \
                --teacher_model Qwen/Qwen3-Embedding-4B \
                --data_size 794554 \
                --batch_size 128 \
                --num_epochs 10 \
                --max_length 512 \
                --language eng \
                --get_id_iso \
                --use_pos \
                --mteb_eval \
                --taid_t_start 0.7 \
                --taid_alpha 2e-04 \
                --loss_type "$loss_type" \
                --add_prefix True \
                --distill_weight "$distill_weight" \
                --lr "$lr"
        done
    done
done