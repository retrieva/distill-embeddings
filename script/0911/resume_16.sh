#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=20:00:00
#PJM -j
#PJM -o logs/final/resume_16.log

set -eu

# Ensure working directory is the repo root (distill-embeddings)
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

module load cuda cudnn nccl gcc
nvidia-smi

# Quieter + stable threading
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=false
export LOKY_MAX_CPU_COUNT=32

# SSL for HF downloads
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

# Experiment params (match 16.sh)
STUDENT="nomic-ai/modernbert-embed-base-unsupervised"
TEACHER="Qwen/Qwen3-Embedding-4B"
DATA_SIZE=1794545
DATA_NAME="gte_plus"
BATCH_SIZE=128
EPOCHS=3
MAXLEN=512
LANG="eng"
LOSS="ckd"
LR="1e-4"

# Optional: allow manual override as first arg
if [ "${1:-}" != "" ]; then
  CKPT_OVERRIDE="$1"
else
  CKPT_OVERRIDE=""
fi

# Locate last.ckpt robustly
BASE="output/result/${STUDENT//\//_}/${TEACHER//\//_}/${DATA_SIZE}"
if [ -n "${CKPT_OVERRIDE}" ]; then
  CKPT="${CKPT_OVERRIDE}"
else
  CKPT=$(ls -1t ${BASE}/${DATA_NAME}_e${EPOCHS}_bs*_wsd*_${LOSS}*/last.ckpt 2>/dev/null | head -n 1 || true)
fi
if [ -z "${CKPT}" ]; then
  LRNUM=$(uv run python -c "print(float('${LR}'))")
  GBS=$(( BATCH_SIZE * ${WORLD_SIZE:-1} ))
  CKPT="${BASE}/${DATA_NAME}_e${EPOCHS}_bs${GBS}_wsd${LRNUM}_${LOSS}/last.ckpt"
fi
OUT_DIR_BASE="${BASE}/${DATA_NAME}_e${EPOCHS}_bs*_wsd*_${LOSS}*"
STRATEGY_ARGS=""
if [ ! -f "${CKPT}" ]; then
  echo "last.ckpt is not a regular file (Looked at: ${CKPT})"
  echo "Trying to resume from a DeepSpeed-style directory checkpoint (if available)..."
  if [ -d "${CKPT}" ]; then
    DS_DIR="${CKPT}"
    CKPT="${DS_DIR}"
    STRATEGY_ARGS="--strategy deepspeed"
  else
    DS_DIR=$(ls -1dt ${OUT_DIR_BASE}/checkpoints/* 2>/dev/null | head -n 1 || true)
    if [ -z "${DS_DIR}" ]; then
      DS_DIR=$(ls -1dt ${OUT_DIR_BASE}/checkpoints 2>/dev/null | head -n 1 || true)
    fi
  fi
  if [ -z "${DS_DIR}" ]; then
    echo "No DeepSpeed checkpoint directory found either. Cannot resume."
    exit 1
  fi
  if [ -z "${STRATEGY_ARGS}" ]; then
    echo "Found DS checkpoint dir: ${DS_DIR}"
    CKPT="${DS_DIR}"
    STRATEGY_ARGS="--strategy deepspeed"
  fi
  OUT_DIR=$(dirname "${DS_DIR}")
else
  OUT_DIR=$(dirname "${CKPT}")
fi

# Try to infer W&B run id
get_run_id() {
  MET="$1"
  if [ -f "${MET}" ]; then
    uv run python - "$MET" <<'PY'
import json, sys
try:
  print(json.load(open(sys.argv[1])).get("id",""))
except Exception:
  print("")
PY
  else
    echo ""
  fi
}
RUN_ID="$(get_run_id "${OUT_DIR}/wandb/latest-run/files/wandb-metadata.json")"
if [ -z "${RUN_ID}" ]; then
  META_FALLBACK=$(ls -1t "${OUT_DIR}"/wandb/run-*/files/wandb-metadata.json 2>/dev/null | head -n 1 || true)
  RUN_ID="$(get_run_id "${META_FALLBACK:-/nonexistent}")"
fi

echo "Resuming 16 (ckd) from: ${CKPT} (run_id=${RUN_ID:-none})"
uv run python -m src.training.train \
  --student_model "${STUDENT}" \
  --teacher_model "${TEACHER}" \
  --data_size "${DATA_SIZE}" \
  --data_name "${DATA_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${EPOCHS}" \
  --max_length "${MAXLEN}" \
  --language "${LANG}" \
  --get_id_iso \
  --mteb_eval \
  --loss_type "${LOSS}" \
  --add_prefix False \
  --gradient_checkpointing False \
  --distill_weight 1.0 \
  --lr "${LR}" \
  ${STRATEGY_ARGS} \
  $( [ -e "${CKPT:-/nonexistent}" ] && printf %s "--ckpt_path ${CKPT}" ) \
  $( [ -n "${RUN_ID}" ] && printf %s "--your_run_id ${RUN_ID}" )
