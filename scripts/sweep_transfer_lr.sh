#!/usr/bin/env bash
# Sweeps a small grid of (learning_rate, max_steps) combinations for
# `swsearch train-transfer`, evaluating each against the same baseline
# comparison used throughout this project's transfer-learning work.
#
# Disk-conscious by design: only the trained model weights (~90MB each) are
# kept per combination. Embeddings + the FAISS index (the expensive,
#100GB+ part) are written to one shared, reused location that gets
# overwritten every iteration, since re-embedding the whole corpus per
# combination and keeping every copy around isn't affordable at this
# corpus's scale (a single combination's embeddings + index was ~127GB
# earlier in this project).
#
# warmup_ratio and freeze_layers are held at train-transfer's defaults for
# this sweep (varying every knob at once turns this into a combinatorial
# explosion of multi-hour runs); adjust SWEEP below or pass them through if
# you want to sweep those too.
#
# Usage: bash scripts/sweep_transfer_lr.sh
# Run from the repo root with the venv already activated.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SWEEP_DIR="data/processed/models/transfer_learning/sweep"
SHARED_EMBEDDINGS="$SWEEP_DIR/_shared_embeddings"
SHARED_META_DB="$SWEEP_DIR/_shared_paragraphs.index.meta.db"
SHARED_INDEX="$SWEEP_DIR/_shared_paragraphs.index"
SUMMARY="$SWEEP_DIR/summary_$(date +%Y%m%d_%H%M%S).txt"
LOG="data/pipeline_sweep_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$SWEEP_DIR"
: > "$SUMMARY"

# "learning_rate:max_steps" pairs, bracketing the known-good point
# (5e-6, 8000) from earlier in this project in both directions.
SWEEP=(
  "1e-6:8000"
  "5e-6:4000"
  "5e-6:8000"
  "5e-6:12000"
  "1e-5:8000"
)

{
  for combo in "${SWEEP[@]}"; do
    lr="${combo%%:*}"
    steps="${combo##*:}"
    tag="lr${lr}_steps${steps}"
    model_dir="$SWEEP_DIR/$tag/model"

    echo "=================================================="
    echo "=== Sweep combo: learning_rate=$lr max_steps=$steps ($tag)"
    echo "=================================================="

    rm -rf "$model_dir" "$SHARED_EMBEDDINGS" "$SHARED_META_DB" "$SHARED_INDEX"

    swsearch train-transfer --output-dir "$model_dir" \
      --learning-rate "$lr" --max-steps "$steps"

    swsearch embed --model-name "$model_dir" \
      --output-dir "$SHARED_EMBEDDINGS" \
      --meta-db "$SHARED_META_DB"

    swsearch build-index --index-type ivfpq \
      --embeddings-dir "$SHARED_EMBEDDINGS" \
      --meta-db "$SHARED_META_DB" \
      --index-path "$SHARED_INDEX"

    echo "--- Results for $tag ---" | tee -a "$SUMMARY"
    swsearch evaluate --index-path "$SHARED_INDEX" \
      --meta-db "$SHARED_META_DB" \
      --model-name "$model_dir" | tee -a "$SUMMARY"
    echo "" >> "$SUMMARY"

    # Free the shared embeddings/index before the next combo -- keep only
    # this combo's small model weights.
    rm -rf "$SHARED_EMBEDDINGS" "$SHARED_META_DB" "$SHARED_INDEX"
  done

  echo "=================================================="
  echo "=== Sweep complete. Summary: $SUMMARY"
  echo "=================================================="
  cat "$SUMMARY"
} 2>&1 | tee -a "$LOG"
