#!/usr/bin/env bash
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=20:00:00
module load cuda cudnn nccl gcc

nvidia-smi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")

set -euo pipefail

# Batch re-run MTEB and update W&B summary for many experiments.
# Defaults match training layout: output/result/<student>/<teacher>/<data_size>/<code_name>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "${REPO_ROOT}"

# Default to the most commonly used pair under output/result
ROOT=${ROOT:-"output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B"}
# Under this ROOT, experiments live at <data_size>/<code_name>
PATTERN=${PATTERN:-"1794545/*"}
# Optional: base student model (HF id). If unset, infer from ROOT's first segment
STUDENT=${STUDENT:-}
EPOCH=${EPOCH:-}
BENCHMARK=${BENCHMARK:-"MTEB(eng, v2)"}
LANGUAGE=${LANGUAGE:-}
BATCH_SIZE=${BATCH_SIZE:-}
NUM_WORKERS=${NUM_WORKERS:-}
ADD_PREFIX=${ADD_PREFIX:-}
PROJECT=${PROJECT:-"distillation"}
REUSE_CACHED=${REUSE_CACHED:-}
CACHED_ONLY=${CACHED_ONLY:-}
ENTITY=${ENTITY:-}
RESUME_MODE=${RESUME_MODE:-}

usage() {
  cat <<USAGE
Usage: ROOT=<root> PATTERN='<glob>' ./script/posthoc_eval_all.sh

Env vars:
  ROOT         Root directory of experiments (default: output/result/nomic-ai_modernbert-embed-base-unsupervised/Qwen_Qwen3-Embedding-4B)
  PATTERN      Glob under ROOT to locate experiment dirs (default: 1794545/*)
  EPOCH        If set, use checkpoints/<EPOCH>.ckpt instead of last.ckpt
  BENCHMARK    Benchmark alias/name (default: MTEB(eng, v2))
  LANGUAGE     Override language (optional)
  BATCH_SIZE   Override encode batch size (optional)
  NUM_WORKERS  Override num_workers for evaluation (optional)
  ADD_PREFIX   Force add_prefix true|false (optional)
  PROJECT      W&B project (default: distillation)
  STUDENT      Base student model HF id (e.g., nomic-ai/modernbert-embed-base-unsupervised)
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage; exit 0
fi

extra=()
[[ -n "$LANGUAGE" ]] && extra+=(--language "$LANGUAGE")
[[ -n "$BATCH_SIZE" ]] && extra+=(--batch_size "$BATCH_SIZE")
[[ -n "$NUM_WORKERS" ]] && extra+=(--num_workers "$NUM_WORKERS")
[[ -n "$ADD_PREFIX" ]] && extra+=(--add_prefix "$ADD_PREFIX")
[[ -n "$EPOCH" ]] && extra+=(--epoch "$EPOCH")
[[ -n "$REUSE_CACHED" ]] && extra+=(--reuse_cached)
[[ -n "$CACHED_ONLY" ]] && extra+=(--cached_only)
[[ -n "$ENTITY" ]] && extra+=(--entity "$ENTITY")
[[ -n "$RESUME_MODE" ]] && extra+=(--resume_mode "$RESUME_MODE")

# If STUDENT not set, try inferring from ROOT's first path segment under output/result by replacing first '_' with '/'
if [[ -z "$STUDENT" ]]; then
  first_seg="$(echo "$ROOT" | sed -E 's#^output/result/([^/]+)/.*#\1#')"
  if [[ "$first_seg" != "$ROOT" ]]; then
    STUDENT="${first_seg/_//}"
  fi
fi
[[ -n "$STUDENT" ]] && extra+=(--student_model "$STUDENT")

uv run python -m src.evaluation.posthoc_eval_batch \
  --root "$ROOT" \
  --pattern "$PATTERN" \
  --benchmark_name "$BENCHMARK" \
  --project "$PROJECT" \
  "${extra[@]}"
