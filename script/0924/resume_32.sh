#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=10:00:00
#PJM -j
#PJM -o logs/0924/resume_32.log

set -eu

# Ensure we run from the repository root even when pjsub stages a copy of the script
if [ -n "${PJM_O_WORKDIR:-}" ]; then
  REPO_ROOT_CANDIDATE="${PJM_O_WORKDIR}"
  if REPO_ROOT=$(cd "${REPO_ROOT_CANDIDATE}" && git rev-parse --show-toplevel 2>/dev/null); then
    :
  else
    REPO_ROOT="${REPO_ROOT_CANDIDATE}"
  fi
else
  SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
  if REPO_ROOT=$(cd "${SCRIPT_DIR}" && git rev-parse --show-toplevel 2>/dev/null); then
    :
  else
    REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
  fi
fi
cd "${REPO_ROOT}"

module load cuda cudnn nccl gcc
nvidia-smi

# Per-job cache isolation to avoid Triton/Inductor cache conflicts
# Detect a job id from common schedulers, fallback to PID
JOB_ID="${PJM_JOBID:-${SLURM_JOB_ID:-${PBS_JOBID:-$$}}}"
CACHE_BASE="${TMPDIR:-/tmp}/de-cache/${USER}/${JOB_ID}"
export TRITON_CACHE_DIR="${CACHE_BASE}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_BASE}/inductor"
export XDG_CACHE_HOME="${CACHE_BASE}/xdg"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${XDG_CACHE_HOME}" || true
echo "Using TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "Using TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}"
trap 'rm -rf "${CACHE_BASE}" || true' EXIT

# Quieter + stable threading
export OPENBLAS_NUM_THREADS=32
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export TOKENIZERS_PARALLELISM=false
export LOKY_MAX_CPU_COUNT=32

# SSL for HF downloads (if certifi is available)
if SSL_CERT_PATH=$(uv run python -c "import certifi; print(certifi.where())" 2>/dev/null); then
  export SSL_CERT_FILE="${SSL_CERT_PATH}"
else
  echo "Warning: certifi not available; continuing without SSL_CERT_FILE override" >&2
fi

# Experiment params (match 16.sh)
STUDENT="nomic-ai/modernbert-embed-base-unsupervised"
TEACHER="Qwen/Qwen3-Embedding-4B"
DATA_SIZE=1794545
DATA_NAME="fineweb"
BATCH_SIZE=128
EPOCHS=3
MAXLEN=512
LANG="eng"
LOSS="infocse"
LR="1e-4"
ADD_PREFIX=false

# Optional: allow manual override as first arg
if [ "${1:-}" != "" ]; then
  CKPT_OVERRIDE="$1"
else
  CKPT_OVERRIDE=""
fi

# Locate checkpoint defaults
BASE="output/result/${STUDENT//\//_}/${TEACHER//\//_}/${DATA_SIZE}"
LRNUM=$(uv run python -c "print(float('${LR}'))")
GBS=$(( BATCH_SIZE * ${WORLD_SIZE:-1} ))
DEFAULT_DIR="${BASE}/${DATA_NAME}_e${EPOCHS}_bs${GBS}_wsd${LRNUM}_${LOSS}"
DEFAULT_FILE_CKPT="${DEFAULT_DIR}/checkpoints/epoch=01.ckpt"

if [ -n "${CKPT_OVERRIDE}" ]; then
  CKPT="${CKPT_OVERRIDE}"
else
  CKPT="${DEFAULT_FILE_CKPT}"
fi

if [ -z "${CKPT}" ]; then
  CKPT="${DEFAULT_DIR}/last.ckpt"
fi
OUT_DIR_BASE="${BASE}/${DATA_NAME}_e${EPOCHS}_bs*_wsd*_${LOSS}*"
STRATEGY_ARGS=""
if [ ! -e "${CKPT}" ]; then
  echo "Specified checkpoint does not exist (Looked at: ${CKPT})"
  echo "Searching for an explicit checkpoint file or directory..."
  if [ -d "${CKPT}" ]; then
    :
  else
    CAND_CKPT=$(ls -1dt ${OUT_DIR_BASE}/checkpoints/* 2>/dev/null | head -n 1 || true)
    if [ -z "${CAND_CKPT}" ]; then
      CAND_CKPT=$(ls -1dt ${OUT_DIR_BASE}/checkpoints 2>/dev/null | head -n 1 || true)
    fi
    CKPT="${CAND_CKPT}"
  fi
  if [ -z "${CKPT}" ]; then
    echo "No checkpoint file or directory found. Cannot resume."
    exit 1
  fi
  if [ -d "${CKPT}" ]; then
    echo "Found DeepSpeed checkpoint dir: ${CKPT}"
    STRATEGY_ARGS="--strategy deepspeed"
  elif [ -f "${CKPT}" ]; then
    echo "Found checkpoint file: ${CKPT}"
  else
    echo "Checkpoint path is neither a file nor a directory: ${CKPT}"
    exit 1
  fi
fi

OUT_DIR=$(dirname "${CKPT}")
if [ "$(basename "${OUT_DIR}")" = "checkpoints" ]; then
  OUT_DIR=$(dirname "${OUT_DIR}")
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

echo "Resuming 32 (infocse) from: ${CKPT} (run_id=${RUN_ID:-none})"
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
