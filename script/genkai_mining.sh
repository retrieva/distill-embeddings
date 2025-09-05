#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=07:00:00
#PJM -j 

module load cuda cudnn nccl gcc

# GPU情報を表示
nvidia-smi

uv sync

uv run python src/data_processing/gte_en_plus_data/mining_fever_neg.py 