#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASETS=("tlcodesum" "funcom")
STAGES=("retriever" "retrieve" "generator" "evaluate")
TOP_K=5
RETRIEVER="codebert"

usage() {
  cat <<'EOF'
Usage:
  bash run_all.sh
  bash run_all.sh --dataset tlcodesum
  bash run_all.sh --dataset funcom --retriever simcse--top-k 3
  bash run_all.sh --retriever simcse
  bash run_all.sh --stage retriever
  bash run_all.sh --dataset tlcodesum  --top-k 3 --retriever codebert
  bash run_all.sh --top-k 0

Options:
  --dataset <name>   One of: tlcodesum, funcom, all. Default: all
  --stage <name>     One of: retriever, retrieve, generator, evaluate, all. Default: all
  --top-k <int>      Retrieved comment count. Default: 5
  --retriever <name> One of: codebert, simcse. Default: simcse
  --help             Show this message
EOF
}

SELECTED_DATASET="all"
SELECTED_STAGE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      SELECTED_DATASET="${2:-}"
      shift 2
      ;;
    --stage)
      SELECTED_STAGE="${2:-}"
      shift 2
      ;;
    --top-k)
      TOP_K="${2:-}"
      shift 2
      ;;
    --retriever)
      RETRIEVER="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$SELECTED_DATASET" != "all" && "$SELECTED_DATASET" != "tlcodesum" && "$SELECTED_DATASET" != "funcom" ]]; then
  echo "Invalid --dataset: $SELECTED_DATASET" >&2
  exit 1
fi

if [[ "$SELECTED_STAGE" != "all" && "$SELECTED_STAGE" != "retriever" && "$SELECTED_STAGE" != "retrieve" && "$SELECTED_STAGE" != "generator" && "$SELECTED_STAGE" != "evaluate" ]]; then
  echo "Invalid --stage: $SELECTED_STAGE" >&2
  exit 1
fi

if [[ "$RETRIEVER" != "codebert" && "$RETRIEVER" != "simcse" ]]; then
  echo "Invalid --retriever: $RETRIEVER" >&2
  exit 1
fi

if ! [[ "$TOP_K" =~ ^[0-9]+$ ]]; then
  echo "Invalid --top-k: $TOP_K" >&2
  exit 1
fi

run_stage() {
  local dataset="$1"
  local stage="$2"

  case "$stage" in
    retriever)
      echo "[${dataset}] train retrieval encoder (retriever=${RETRIEVER})"
      python train_retrieval_encoder.py --dataset "$dataset" --retriever "$RETRIEVER"
      ;;
    retrieve)
      echo "[${dataset}] build retrieved comments (top_k=${TOP_K}, retriever=${RETRIEVER})"
      python build_retrieved_comments.py --dataset "$dataset" --top-k "$TOP_K" --retriever "$RETRIEVER"
      ;;
    generator)
      echo "[${dataset}] train comment generator (top_k=${TOP_K}, retriever=${RETRIEVER})"
      python train_comment_generator.py --dataset "$dataset" --top-k "$TOP_K" --retriever "$RETRIEVER"
      ;;
    evaluate)
      echo "[${dataset}] evaluate comment generator (top_k=${TOP_K}, retriever=${RETRIEVER})"
      python evaluate_comment_generator.py --dataset "$dataset" --top-k "$TOP_K" --retriever "$RETRIEVER"
      ;;
    *)
      echo "Unsupported stage: $stage" >&2
      exit 1
      ;;
  esac
}

for dataset in "${DATASETS[@]}"; do
  if [[ "$SELECTED_DATASET" != "all" && "$SELECTED_DATASET" != "$dataset" ]]; then
    continue
  fi

  for stage in "${STAGES[@]}"; do
    if [[ "$SELECTED_STAGE" != "all" && "$SELECTED_STAGE" != "$stage" ]]; then
      continue
    fi
    run_stage "$dataset" "$stage"
  done
done

echo "All requested runs completed."
