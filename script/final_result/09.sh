#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=10:00:00
#PJM -j
#PJM -o logs/final/09.log

module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

# Per-job cache isolation to avoid Triton/Inductor cache conflicts
# Detect a job id from common schedulers, fallback to PID
JOB_ID="${PJM_JOBID:-${SLURM_JOB_ID:-${PBS_JOBID:-$$}}}"
CACHE_BASE="${TMPDIR:-/tmp}/de-cache/${USER}/${JOB_ID}"
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
# TorchInductor typically uses XDG_CACHE_HOME, but many builds also respect TORCHINDUCTOR_CACHE_DIR.
# We set a dedicated dir; if not honored, XDG_CACHE_HOME can be used instead (see docs).
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/inductor"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" || true
echo "Using TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "Using TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"


for loss_type in "kld"; do
    for lr in 1e-4; do
        for distill_weight in 0.98;do
            uv run python -m src.training.train \
                --student_model nomic-ai/modernbert-embed-base-unsupervised \
                --teacher_model Qwen/Qwen3-Embedding-4B \
                --data_size 1794550 \
                --data_name gte_en_plus_w_neg \
                --batch_size 128 \
                --num_epochs 3 \
                --max_length 512 \
                --language eng \
                --get_id_iso \
                --use_pos \
                --use_neg \
                --mteb_eval \
                --taid_t_start 0.7 \
                --taid_alpha 5e-04 \
                --loss_type "$loss_type" \
                --add_prefix True \
                --gradient_checkpointing False \
                --distill_weight "$distill_weight" \
                --lr "$lr"
        done
    done
done
