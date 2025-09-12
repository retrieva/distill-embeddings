#!/usr/bin/env bash
set -euo pipefail
module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")
# Simple wrapper to run MTEB eval with consistent column ordering
# and optional task drops to match wandb_agg outputs.

# Config
LANGUAGE=${LANGUAGE:-eng}
BENCHMARK_NAME=${BENCHMARK_NAME:-"MTEB(eng, v2)"}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-4}
ADD_PREFIX=${ADD_PREFIX:-true}
TASK_SORT=${TASK_SORT:-category_alpha}   # category_alpha | alpha | none
# Tasks to drop from outputs (comma-separated)
DROP_TASKS=${DROP_TASKS:-"Touche2020Retrieval.v3,ToxicConversationsClassification,TweetSentimentExtractionClassification,TwentyNewsgroupsClustering.v2,TwitterSemEval2015,TwitterURLCorpus"}

models=("nomic-ai/modernbert-embed-base-unsupervised")
for model_name in "${models[@]}"; do
    echo "Running MTEB for: ${model_name} (lang=${LANGUAGE}, bench=${BENCHMARK_NAME})"
    uv run python -m src.evaluation.run_mteb \
        --model_name "${model_name}" \
        --language "${LANGUAGE}" \
        --benchmark_name "${BENCHMARK_NAME}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --add_prefix "${ADD_PREFIX}" \
        --wandb_style \
        --task_sort "${TASK_SORT}" \
        --drop_tasks "${DROP_TASKS}"
done
