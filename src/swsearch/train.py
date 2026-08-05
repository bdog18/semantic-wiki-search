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

from swsearch.config import settings
from swsearch.logutil import get_logger

logger = get_logger(__name__)


def load_triplet_dataset(triplets_dir: str):
    """Stream mined (anchor, positive, negative) triplets from JSONL files
    without loading the whole corpus into memory -- mining a full enwiki
    corpus can produce tens of millions of triplets, the same scale that
    OOM'd a flat FAISS index earlier this project. Extra columns the miner
    writes (source, url) are dropped; TripletLoss only wants these three.
    """
    from datasets import load_dataset

    files = sorted(glob.glob(os.path.join(triplets_dir, "*.jsonl")))
    if not files:
        raise ValueError(f"No triplet JSONL files found in {triplets_dir}; run `swsearch mine-triplets` first.")

    dataset = load_dataset("json", data_files=files, split="train", streaming=True)
    return dataset.select_columns(["anchor", "positive", "negative"])


def train_transfer_model(
    triplets_dir: str,
    output_dir: str,
    base_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    max_steps: int = 20000,
    learning_rate: float = 2e-5,
    margin: float = 5.0,
    device: str | None = None,
) -> None:
    """Fine-tune base_model_name on triplets mined from the baseline index
    with TripletLoss, saving the result to output_dir. That path can then be
    passed straight to `swsearch embed --model-name <output_dir>` -- sentence-
    transformers loads a local folder the same way it loads a hub model name.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = SentenceTransformer(base_model_name, device=device or settings.model.device)
    train_dataset = load_triplet_dataset(triplets_dir)
    loss = losses.TripletLoss(model=model, triplet_margin=margin)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=100,
        save_strategy="steps",
        save_steps=max(max_steps // 5, 1),
        save_total_limit=2,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train_dataset, loss=loss)

    logger.info("Fine-tuning %s on triplets from %s for %d steps...", base_model_name, triplets_dir, max_steps)
    trainer.train()

    model.save(output_dir)
    logger.info("Fine-tuned model saved to %s", output_dir)
