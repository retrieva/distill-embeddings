#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L node=1
#PJM -L elapse=24:00:00
#PJM -j
#PJM -o logs/0916/genkai_main03.log

module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

# Simple per-GPU lock queue so jobs sharing GPUs run sequentially
GPU_LOCK_DIR=${GPU_LOCK_DIR:-/tmp/gpu_locks}
mkdir -p "$GPU_LOCK_DIR"

# Track background job PIDs to aggregate exit statuses
JOB_PIDS=""

run_job() {
  devices="$1"  # e.g., "6" or "0,1" or "3,4,5,6"
  log_file="$2"
  shift 2
  (
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
    cmd_str="$*"
    ts() { date '+%F %T'; }
    echo "[$(ts)] START  GPUs=[$devices] -> $cmd_str (log=$log_file)"
    nohup "$@" > "$log_file" 2>&1 &
    child=$!
    echo "[$(ts)] RUNNING GPUs=[$devices] pid=$child"
    set +e
    wait "$child"
    status=$?
    echo "[$(ts)] END    GPUs=[$devices] pid=$child status=$status (log=$log_file)"
    # Propagate child exit status to this subshell so the parent can detect failures
    exit "$status"
  ) &
  # Record the subshell PID so we can wait on it later
  JOB_PIDS="$JOB_PIDS $!"
}

# Launch jobs concurrently; GPU conflicts serialize via locks
run_job "7" logs/final/28.log bash script/final_result/28.sh
run_job "6" logs/final/29.log bash script/final_result/29.sh
run_job "5" logs/final/30.log bash script/final_result/30.sh
run_job "4" logs/final/31.log bash script/final_result/31.sh
run_job "3" logs/final/32.log bash script/final_result/32.sh
run_job "1" logs/final/33.log bash script/final_result/33.sh
run_job "0" logs/final/33.log bash script/final_result/34.sh

# Wait for all jobs and exit non-zero if any failed
FAIL=0
for pid in $JOB_PIDS; do
  if ! wait "$pid"; then
    FAIL=1
  fi
done
exit "$FAIL"