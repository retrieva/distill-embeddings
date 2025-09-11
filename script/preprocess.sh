#!/bin/bash

uv run python src/data_processing/finweb/encode.py \
    --data_name "hotchpotch/fineweb-2-edu-japanese" \
    --subset_name "fineweb" \
    --sample_size 1794545 \
    --output_dir "data" \
    --teacher_model "Qwen/Qwen3-Embedding-4B" \
    --long_batch_size 2 \
    --short_batch_size 64 \
    --max_length 4096
