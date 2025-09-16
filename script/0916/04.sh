#!/usr/bin/env bash
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=20:00:00
#PJM -j
#PJM -o logs/0916/04.log


module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

set -euo pipefail


uv run python -m src.evaluation.posthoc_eval_to_wandb \
    --ckpt 'output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B/1794545/gte_plus_e3_bs128_wsd0.0001_mse/checkpoints/epoch=02.ckpt' \
    --run_id "yw8ixh6x" \
    --benchmark_name "MTEB(eng, v2)" \
    --project distillation