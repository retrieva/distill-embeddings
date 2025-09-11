#!/usr/bin/env bash
set -euo pipefail

# Batch re-run MTEB and update W&B summary for many experiments.
# Defaults match training layout: output/result/<student>/<teacher>/<data_size>/<code_name>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
cd "${REPO_ROOT}"

ROOT=${ROOT:-"output/result"}
PATTERN=${PATTERN:-"*/*/*/*"}
EPOCH=${EPOCH:-}
BENCHMARK=${BENCHMARK:-"MTEB(eng, v2)"}
LANGUAGE=${LANGUAGE:-}
BATCH_SIZE=${BATCH_SIZE:-}
NUM_WORKERS=${NUM_WORKERS:-}
ADD_PREFIX=${ADD_PREFIX:-}
PROJECT=${PROJECT:-"distillation"}

usage() {
  cat <<USAGE
Usage: ROOT=output/result PATTERN='*/*/*/*' ./script/posthoc_eval_all.sh

Env vars:
  ROOT         Root directory of experiments (default: output/result)
  PATTERN      Glob under ROOT to locate experiment dirs (default: */*/*/*)
  EPOCH        If set, use checkpoints/<EPOCH>.ckpt instead of last.ckpt
  BENCHMARK    Benchmark alias/name (default: MTEB(eng, v2))
  LANGUAGE     Override language (optional)
  BATCH_SIZE   Override encode batch size (optional)
  NUM_WORKERS  Override num_workers for evaluation (optional)
  ADD_PREFIX   Force add_prefix true|false (optional)
  PROJECT      W&B project (default: distillation)
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

python -m src.evaluation.posthoc_eval_batch \
  --root "$ROOT" \
  --pattern "$PATTERN" \
  --benchmark_name "$BENCHMARK" \
  --project "$PROJECT" \
  "${extra[@]}"
