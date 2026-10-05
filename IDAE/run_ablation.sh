#!/usr/bin/env bash
  # bash run_ablation.sh contrastive 
  # bash run_ablation.sh main  
  # bash run_ablation.sh rand_ex
  # bash run_ablation.sh token
  # bash run_ablation.sh semantic


set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$ROOT_DIR/src"

cd "$SRC_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ABLATION="${1:-main}"

case "$ABLATION" in
  contrastive|main|rand_ex|token|semantic|full|wo_cl)
    ;;
  *)
    echo "Unsupported ablation: $ABLATION" >&2
    echo "Usage: bash run_ablation.sh [main|contrastive|rand_ex|token|semantic]" >&2
    exit 1
    ;;
esac

export IDAE_ABLATION="$ABLATION"

echo "Running ablation: $IDAE_ABLATION"
echo "Step 1/4: build train exemplars"
python build_nearest_examples_train.py

echo "Step 2/4: build valid/test exemplars"
python build_nearest_examples_test.py

echo "Step 3/4: train generator"
python codet5_dec.py

echo "Step 4/4: evaluate on test set"
python evaluate.py

echo "Completed ablation: $IDAE_ABLATION"
