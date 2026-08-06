"""Fine-tune a pretrained SentenceTransformer on mined triplets ("transfer
learning" in the baseline / transfer-learning / custom-transformer 3-model
comparison -- see README.md).

Reuses the single triplets file mined once against the baseline index
(`swsearch mine-triplets`); no separate mining run is needed per target
model, since triplet mining's hard negatives only depend on the embedding
space used to find them, not on the model being trained.
"""
import glob
import os

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator

from swsearch.config import settings
from swsearch.logutil import get_logger

logger = get_logger(__name__)


def load_triplet_dataset(triplets_dir: str, holdout_files: int = 1):
    """Stream mined (anchor, positive, negative) triplets from JSONL files
    without loading the whole corpus into memory -- mining a full enwiki
    corpus can produce tens of millions of triplets, the same scale that
    OOM'd a flat FAISS index earlier this project. Extra columns the miner
    writes (source, url) are dropped; the loss/evaluator only want these
    three.

    Reserves holdout_files whole files as a small in-memory eval split (for
    TripletEvaluator/load_best_model_at_end -- see train_transfer_model) and
    streams the rest as the training set. Returns (train_dataset,
    eval_dataset), where eval_dataset is None if there aren't enough files to
    hold any out.
    """
    from datasets import load_dataset

    files = sorted(glob.glob(os.path.join(triplets_dir, "*.jsonl")))
    if not files:
        raise ValueError(f"No triplet JSONL files found in {triplets_dir}; run `swsearch mine-triplets` first.")

    columns = ["anchor", "positive", "negative"]
    n_holdout = min(holdout_files, max(len(files) - 1, 0))
    eval_files, train_files = files[:n_holdout], files[n_holdout:]

    train_dataset = load_dataset("json", data_files=train_files, split="train", streaming=True).select_columns(columns)
    eval_dataset = None
    if eval_files:
        eval_dataset = load_dataset("json", data_files=eval_files, split="train", streaming=False).select_columns(columns)

    return train_dataset, eval_dataset


def train_transfer_model(
    triplets_dir: str,
    output_dir: str,
    base_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 20000,
    learning_rate: float = 2e-5,
    scale: float = 20.0,
    device: str | None = None,
) -> None:
    """Fine-tune base_model_name on triplets mined from the baseline index
    with MultipleNegativesRankingLoss, saving the best checkpoint (by held-
    out triplet accuracy, not necessarily the last step) to output_dir. That
    path can then be passed straight to `swsearch embed --model-name
    <output_dir>` -- sentence-transformers loads a local folder the same way
    it loads a hub model name.

    Uses MultipleNegativesRankingLoss (in-batch negatives, plus each row's
    mined hard negative as an extra candidate) instead of TripletLoss.
    TripletLoss was tried first and produced a collapsed model: all-MiniLM-
    L6-v2 ends its pipeline with a Normalize module, so embeddings are unit
    vectors even inside the loss, which bounds Euclidean distance to [0, 2].
    TripletLoss's default margin (5.0) is unreachable in that range, so
    `relu(d(a,p) - d(a,n) + margin)` never clips to zero for any triplet --
    every example, however well-separated, keeps contributing full gradient
    for the entire run, which is exactly the kind of unbounded push that
    causes representation collapse. MultipleNegativesRankingLoss has no such
    margin to mis-set (scale is a softmax temperature, not a distance
    bound), and in-batch negatives are far more robust to occasional false
    negatives than always training against the single hardest mined one.

    batch_size defaults small (with gradient_accumulation_steps making up the
    effective batch size) because training -- unlike the inference-only
    encode() calls elsewhere in this project -- has to retain every layer's
    activations for the backward pass. fp16 is enabled on CUDA for the same
    reason mining/triplets.py already calls model.half() for GPU inference --
    it roughly halves activation memory with negligible quality impact for a
    model this size.

    Checkpoints are saved every max_steps // 5 steps; load_best_model_at_end
    picks whichever surviving checkpoint scores highest on a held-out slice
    of the mined triplets (via TripletEvaluator's cosine accuracy), not
    necessarily the final one -- protects against the model quietly getting
    worse over the course of a long run with nothing watching for it.
    """
    os.makedirs(output_dir, exist_ok=True)
    resolved_device = device or settings.model.device
    model = SentenceTransformer(base_model_name, device=resolved_device)
    train_dataset, eval_dataset = load_triplet_dataset(triplets_dir)
    loss = losses.MultipleNegativesRankingLoss(model=model, scale=scale)

    evaluator = None
    if eval_dataset is not None:
        evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
        )

    eval_steps = max(max_steps // 5, 1)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=(resolved_device == "cuda"),
        logging_steps=100,
        eval_strategy="steps" if evaluator is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=5,
        load_best_model_at_end=evaluator is not None,
        metric_for_best_model="eval_cosine_accuracy",
        greater_is_better=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    logger.info("Fine-tuning %s on triplets from %s for %d steps...", base_model_name, triplets_dir, max_steps)
    trainer.train()
    if evaluator is not None:
        logger.info("Best checkpoint: %s (eval_cosine_accuracy=%s)", trainer.state.best_model_checkpoint, trainer.state.best_metric)

    model.save(output_dir)
    logger.info("Fine-tuned model saved to %s", output_dir)
