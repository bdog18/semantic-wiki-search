#!/usr/bin/env bash
# Find where fine-tuning actually peaks, by training a SHORT run and reading
# the in-training retrieval curve.
#
# WHY -- the 8,000-step run with corrected mining selected checkpoint-1600 as
# best and declined monotonically after it:
#     step 1600  MRR@10 0.9280   <- best
#     step 3200         0.9273
#     step 4800         0.9249
#     step 6400         0.9112
#     step 8000         0.9121
# The peak is at or before the first evaluation, i.e. after ~0.3% of the
# mined triplets. This run evaluates every 400 steps instead of every 1600,
# to see whether quality is still rising at 400 and topping out around 1600,
# or already falling from the very first checkpoint. Those two shapes mean
# different things: the first says there's a real (if small) gain to
# capture, the second says fine-tuning is only ever losing ground to the
# pretrained weights.
#
# eval_steps is max_steps//5 (train.py), so --max-steps 2000 gives
# evaluations at 400 / 800 / 1200 / 1600 / 2000.
#
# CAVEAT worth remembering when comparing to the 8,000-step run: the LR
# schedule is linear decay across the whole run, so step 400 here is not the
# same model as step 400 there -- it decays 4x faster and warms up over 200
# steps instead of 800. This finds the best model for a 2,000-step schedule,
# which is what you would actually ship; it is not a clean replay of where
# the longer run peaked.
#
# Reuses the 34M triplets already mined with the corrected recipe -- nothing
# about them changed, so re-mining would just burn another 4+ hours.
#
# By default this ONLY trains (~20 min: ~4 min of steps plus five ~3 min
# retrieval evaluations) and prints the curve. Embedding + indexing the
# winner costs ~6 hours and 61GB, so it is opt-in:
#     ./scripts/probe_short_steps.sh          # train + curve only
#     ./scripts/probe_short_steps.sh --full   # ... then embed, index, evaluate
set -euo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO -- see output above"' ERR

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

BASELINE_INDEX="data/processed/models/baseline/runs/all-MiniLM-L6/paragraphs.index"
BASELINE_META_DB="data/processed/models/baseline/runs/all-MiniLM-L6/paragraphs.index.meta.db"

# A fresh directory: runs/lr5e-6_steps8000 currently holds the best
# fine-tune you have (tied with baseline on raw retrieval) and this probe
# must not overwrite it.
RUN_DIR="data/processed/models/transfer_learning/runs/lr5e-6_steps2000"
MODEL_DIR="$RUN_DIR/model"
LOG="data/pipeline_probe2000_$(date +%Y%m%d_%H%M%S).log"

{
echo "[$(date)] ===== Short-step probe: 2000 steps, eval every 400 ====="
test -d data/processed/triplets/parallel_parts || { echo "ERROR: no mined triplets"; exit 1; }
echo "reusing $(find data/processed/triplets/parallel_parts -name '*.jsonl' | wc -l) mined triplet file(s)"

rm -rf "$RUN_DIR"

echo "[$(date)] Training (lr=5e-6, max_steps=2000, evals at 400/800/1200/1600/2000)..."
swsearch train-transfer \
  --output-dir "$MODEL_DIR" \
  --learning-rate 5e-6 --max-steps 2000 \
  --eval-meta-db "$BASELINE_META_DB"

echo
echo "[$(date)] ===== Retrieval curve (this is the answer you came for) ====="
# Column 8 is cosine-MRR@10; printed as a plain table so the shape is
# readable without opening the CSV.
awk -F, 'NR==1 {printf "%8s  %10s\n","steps","MRR@10"; next} {printf "%8s  %10.4f\n",$2,$8}' \
  "$MODEL_DIR"/eval/*.csv
echo
echo "Reference: the 8000-step run peaked at step 1600 with MRR@10 0.9280."
echo "  still climbing at 2000  -> a longer schedule may be worth another look"
echo "  peak in the middle      -> that step count is the one to ship"
echo "  falling from step 400   -> fine-tuning is losing to the pretrained weights"

if [ "$FULL" -eq 0 ]; then
  echo
  echo "[$(date)] Stopping here (train-only mode). To embed + index + evaluate this checkpoint:"
  echo "    ./scripts/probe_short_steps.sh --full"
  echo "Note that re-running with --full retrains from scratch; it does not resume."
else
  echo
  echo "[$(date)] Embedding the corpus with the selected checkpoint (~5h)..."
  swsearch embed \
    --model-name "$MODEL_DIR" \
    --output-dir "$RUN_DIR/embeddings" \
    --meta-db "$RUN_DIR/paragraphs.index.meta.db"

  echo "[$(date)] Building FAISS index (ivfpq, ~45m)..."
  swsearch build-index \
    --index-type ivfpq \
    --embeddings-dir "$RUN_DIR/embeddings" \
    --meta-db "$RUN_DIR/paragraphs.index.meta.db" \
    --index-path "$RUN_DIR/paragraphs.index"

  echo "[$(date)] Evaluating against baseline..."
  for mode in --no-rerank --rerank; do
    echo "--- probe lr5e-6_steps2000 ($mode) ---"
    swsearch evaluate \
      --index-path "$RUN_DIR/paragraphs.index" \
      --meta-db "$RUN_DIR/paragraphs.index.meta.db" \
      --model-name "$MODEL_DIR" $mode
    echo "--- baseline all-MiniLM-L6-v2 ($mode) ---"
    swsearch evaluate \
      --index-path "$BASELINE_INDEX" \
      --meta-db "$BASELINE_META_DB" \
      --model-name all-MiniLM-L6-v2 $mode
  done

  echo
  echo "Full-index reference numbers (clean corpus, 155 queries, +/- ~0.028 on MRR):"
  echo "  baseline          no-rerank MRR 0.3767 | rerank MRR 0.7947"
  echo "  8000-step (fixed) no-rerank MRR 0.3754 | rerank MRR 0.7444"
fi

echo "[$(date)] Done."
} 2>&1 | tee -a "$LOG"

echo "Log written to $LOG"
