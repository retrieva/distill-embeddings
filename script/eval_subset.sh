#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=5:00:00
#PJM -j
#PJM -o logs/eval-subset.log

set -euo pipefail
module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")
# Run MTEB eval like eval.sh but for selected tasks only

# Config
LANGUAGE=${LANGUAGE:-eng}
BENCHMARK_NAME=${BENCHMARK_NAME:-"MTEB(eng, v2)"}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-4}
ADD_PREFIX=${ADD_PREFIX:-false}
TASK_SORT=${TASK_SORT:-category_alpha}   # category_alpha | alpha | none
# Tasks to drop from outputs (comma-separated)
DROP_TASKS=${DROP_TASKS:-}
# Only these tasks will be evaluated (comma-separated, exact match)
ONLY_TASKS=${ONLY_TASKS:-"Touche2020Retrieval.v3,ToxicConversationsClassification,TweetSentimentExtractionClassification,TwentyNewsgroupsClustering.v2,TwitterSemEval2015,TwitterURLCorpus"}

models=("nomic-ai/modernbert-embed-base-unsupervised")
for model_name in "${models[@]}"; do
    echo "Running MTEB subset for: ${model_name} (lang=${LANGUAGE}, bench=${BENCHMARK_NAME}, only=[${ONLY_TASKS}])"
    uv run python -m src.evaluation.run_mteb \
        --model_name "${model_name}" \
        --language "${LANGUAGE}" \
        --benchmark_name "${BENCHMARK_NAME}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --add_prefix "${ADD_PREFIX}" \
        --wandb_style \
        --task_sort "${TASK_SORT}" \
        --drop_tasks "${DROP_TASKS}" \
        --only_tasks "${ONLY_TASKS}"
done

