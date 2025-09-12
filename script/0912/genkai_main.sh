#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L node=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -o logs/0912/genkai_main.log

module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

# Simple per-GPU lock queue so jobs sharing GPUs run sequentially
GPU_LOCK_DIR=${GPU_LOCK_DIR:-/tmp/gpu_locks}
mkdir -p "$GPU_LOCK_DIR"

run_job() {
  devices="$1"  # e.g., "6" or "0,1" or "3,4,5,6"
  log_file="$2"
  shift 2
  (
    set -e
    mkdir -p "$(dirname "$log_file")"
    LOCKS=""
    trap 'for l in $LOCKS; do rmdir "$l" 2>/dev/null || true; done' EXIT INT TERM
    # Acquire locks in sorted order to avoid deadlocks
    for gpu in $(echo "$devices" | tr ',' '\n' | sort -n | uniq); do
      lock="$GPU_LOCK_DIR/gpu_$gpu.lock"
      # Wait until this GPU becomes free
      while ! mkdir "$lock" 2>/dev/null; do
        sleep 2
      done
      LOCKS="$LOCKS $lock"
    done
    export CUDA_VISIBLE_DEVICES="$devices"
    nohup "$@" > "$log_file" 2>&1
  ) &
}

# Launch jobs concurrently; GPU conflicts serialize via locks
run_job "7" logs/encode_clus.log bash script/genkai_encode_clus_en_single.sh
run_job "0,1" logs/preprocess.log bash script/preprocess.sh

# 以下resumeしたい人たち
run_job "3,4" logs/0912/resume_09.log bash script/0912/resume_09.sh
run_job "5" logs/0912/resume_03.log bash script/0912/resume_03.sh
run_job "6" logs/final/13.log bash script/final_result/13.sh

run_job "3" logs/final/22.log bash script/final_result/22.sh
run_job "5" logs/final/04.log bash script/final_result/04.sh
run_job "4" logs/final/14.log bash script/final_result/14.sh
run_job "3,4,5,6" logs/final/23.log bash script/final_result/23.sh
run_job "3,4,5,6" logs/final/24.log bash script/final_result/24.sh

wait
