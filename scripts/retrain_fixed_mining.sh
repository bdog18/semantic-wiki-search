#!/usr/bin/env bash
# Re-mine triplets with the corrected negative-sampling recipe, then retrain
# and re-evaluate the lr5e-6 transfer model.
#
# WHY THIS EXISTS -- the previous fine-tunes lost to plain baseline because
# the mined training data was inverted: negatives were drawn from the
# anchor's top-10 nearest neighbours, i.e. the most relevant paragraphs in
# the corpus. Measured over 3k triplets, 66% of "negatives" were MORE
# anchor-similar than their own paired positive, so most of the training
# signal taught the model to push relevant content away. That is why the
# 500k-step run scored held-out triplet accuracy 0.89 while real corpus MRR
# fell to 0.16.
#
# Four fixes are in effect here (all in src/, not this script):
#   1. negatives sampled from rank band 1000-2000, not 0-10. Verified on a
#      real mined file: cos(anchor,negative) 0.5546 -> 0.3798, now below
#      cos(anchor,positive) 0.4404; inverted pairs 66.7% -> 38.3%;
#      near-duplicate negatives 11.2% -> 0%.
#   2. sentence_anchor_probability back to 0.0 (all-title anchors, the
#      distribution that was accidentally in force for every run that won)
#   3. checkpoint selection by real retrieval MRR@10, not triplet accuracy
#   4. FAISS parallel_mode=3 on IVF indexes. The default (parallelise over
#      queries) returns corrupted results for multi-query searches on this
#      faiss-cpu 1.15.0 build -- 4 anchors at once came back as one id
#      repeated at distance -0.019 where the true neighbours sit at 0.417.
#      Mining searches a batch at a time, so this silently poisoned every
#      negative it drew. Single-query callers (the search engine, and so
#      `swsearch evaluate`) were never affected.
#
# Steps are back to 8,000 -- NOT the 500k of the last run. Longer runs only
# help once the signal is correct, and this run's job is to establish
# whether it is. Compare against baseline before scaling anything up.
#
# ROUGH TIMINGS (RTX 3070 Ti, 42M-paragraph corpus):
#   mine ~4-8h (much deeper search than before -- 2000 neighbours at
#               nprobe=32 rather than 10 at nprobe=10), train ~20m,
#               embed ~3.5h, index ~30m.
# Budget a full day. A single scratch file mined in ~8s, so the mining
# estimate is extrapolated from one file x 20,013; check progress early
# rather than trusting it.
set -euo pipefail
trap 'echo "[$(date)] FAILED at line $LINENO -- see output above"' ERR

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

# Clean, post-fix baseline. Mining reads this to find negatives, and the
# retrieval evaluator reads its metadata store to build an eval corpus.
BASELINE_INDEX="data/processed/models/baseline/runs/all-MiniLM-L6/paragraphs.index"
BASELINE_META_DB="data/processed/models/baseline/runs/all-MiniLM-L6/paragraphs.index.meta.db"

RUN_DIR="data/processed/models/transfer_learning/runs/lr5e-6_steps8000"
MODEL_DIR="$RUN_DIR/model"
LOG="data/pipeline_fixedmining_$(date +%Y%m%d_%H%M%S).log"

# Mining loads a copy of the FAISS index and a CUDA model per worker; 2 is
# what fits alongside everything else on an 8GB card. Raise if you have room.
MINE_WORKERS=2

{
echo "[$(date)] ===== Retrain with corrected mining ====="
echo "config in effect:"
python - <<'PY'
from swsearch.config import settings
m = settings.mining
print(f"  negative rank band      : {m.negative_rank_min}-{m.negative_rank_max}")
print(f"  negatives sampled/anchor: {m.negative_pool_size}")
print(f"  search nprobe           : {m.negative_search_nprobe}")
print(f"  sentence_anchor_prob    : {m.sentence_anchor_probability}")
PY

echo
echo "[$(date)] Step 0/5: resetting resume markers, clearing old triplets and the previous (collapsed 500k) run..."
# CAREFUL: mine-triplets marks a file done with os.replace(f, f + ".completed")
# -- a RENAME, not a sidecar marker. The .completed files ARE the corpus, so
# deleting them destroys data/processed/wikidata_jsonl outright. To re-mine,
# rename them back. Two cases, because `swsearch extract` writes fresh .jsonl
# at the original names without clearing old .completed renames:
#   - a .jsonl already exists -> that one is the fresher extract output, and
#     the .completed alongside it is a stale pre-fix copy, so drop it
#   - no .jsonl exists        -> the .completed IS the current corpus file;
#     rename it back so mining sees it again
restored=0; dropped=0
while IFS= read -r -d '' f; do
  orig="${f%.completed}"
  if [ -e "$orig" ]; then rm -f "$f"; dropped=$((dropped+1))
  else mv "$f" "$orig"; restored=$((restored+1)); fi
done < <(find data/processed/wikidata_jsonl -name "*.jsonl.completed" -print0)
echo "  restored $restored file(s) for re-mining, dropped $dropped stale duplicate(s)"
test "$(find data/processed/wikidata_jsonl -name '*.jsonl' | head -1)" != "" \
  || { echo "ERROR: no .jsonl files to mine -- aborting before wasting a run"; exit 1; }

rm -rf data/processed/triplets/parallel_parts/*
rm -rf "$RUN_DIR"

echo "[$(date)] Step 1/5: mining triplets with the corrected negative sampling..."
swsearch mine-triplets \
  --index-path "$BASELINE_INDEX" \
  --meta-db "$BASELINE_META_DB" \
  --num-workers "$MINE_WORKERS"

echo "[$(date)] Step 2/5: training (lr=5e-6, 8000 steps, checkpoints selected on real retrieval MRR@10)..."
swsearch train-transfer \
  --output-dir "$MODEL_DIR" \
  --learning-rate 5e-6 --max-steps 8000 \
  --eval-meta-db "$BASELINE_META_DB"

echo "[$(date)] Step 3/5: embedding the corpus with the new checkpoint..."
swsearch embed \
  --model-name "$MODEL_DIR" \
  --output-dir "$RUN_DIR/embeddings" \
  --meta-db "$RUN_DIR/paragraphs.index.meta.db"

echo "[$(date)] Step 4/5: building FAISS index (ivfpq)..."
swsearch build-index \
  --index-type ivfpq \
  --embeddings-dir "$RUN_DIR/embeddings" \
  --meta-db "$RUN_DIR/paragraphs.index.meta.db" \
  --index-path "$RUN_DIR/paragraphs.index"

echo "[$(date)] Step 5/5: evaluating -- fine-tune vs baseline, reranked and raw..."
for mode in --no-rerank --rerank; do
  echo "--- fine-tuned lr5e-6_steps8000 ($mode) ---"
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
echo "[$(date)] Done. Model + index at $RUN_DIR"
echo "Reference numbers to beat (post-fix corpus):"
echo "  baseline   no-rerank MRR 0.3767 | rerank MRR 0.7947"
echo "  prior 8k   no-rerank MRR 0.3409 (old inverted mining)"
echo "  prior 500k no-rerank MRR 0.1602 (old inverted mining, overtrained)"
} 2>&1 | tee -a "$LOG"

echo "Log written to $LOG"

# Run directly in your terminal to watch progress:
#     ./scripts/retrain_fixed_mining.sh
