#!/bin/bash

uv run python src/preprocess/data_encode.py \
    --data_name "hotchpotch/fineweb-2-edu-japanese" \
    --subset_name "sample_10BT" \
    --sample_size 50 \
    --output_dir "data" \
    --teacher_model "Qwen/Qwen3-Embedding-4B" \
    --batch_size 8 \
    --max_length 8192