#!/bin/sh
#PJM --batch
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=2:00:00
module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

uv sync

uv run python get_id_iso.py \
    --model_name "Qwen/Qwen3-Embedding-4B"

uv run python get_id_iso.py \
    --model_name "output/result/answerdotai_ModernBERT-base/Qwen_Qwen3-Embedding-4B/1000000/e10_bs128_wsd5e-05_kld_w-pos/checkpoints/epoch=09.ckpt/consolidated_model.pt/pytorch_model.bin"

uv run python get_id_iso.py \
    --model_name "output/result/answerdotai_ModernBERT-base/Qwen_Qwen3-Embedding-4B/1000000/e4_bs128_wsd5e-05_kld_w-pos/checkpoints/epoch=03.ckpt/consolidated_model.pt/pytorch_model.bin"

uv run python get_id_iso.py \
    --model_name "Qwen/Qwen3-Embedding-0.6B"