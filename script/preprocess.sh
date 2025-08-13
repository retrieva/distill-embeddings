#!/bin/bash

uv run python src/preprocess/fineweb_encode.py \
    --data_name "hotchpotch/fineweb-2-edu-japanese" \
    --subset_name "sample_10BT" \
    --sample_size 1000000 \
    --output_dir "data" \
    --teacher_model "Qwen/Qwen3-Embedding-4B" \
    --long_batch_size 1 \
    --short_batch_size 16 \
    --max_length 4096