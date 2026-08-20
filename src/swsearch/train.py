"""Fine-tune a pretrained SentenceTransformer on mined triplets.

This is the experiment README.md documents: it does not beat the
off-the-shelf baseline, and the measurements explaining why live there.

Reuses the single triplets file mined once against the baseline index
(`swsearch mine-triplets`); no separate mining run is needed per target
model, since triplet mining's hard negatives only depend on the embedding
space used to find them, not on the model being trained.
"""
import glob
import json
import os
import random

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator, TripletEvaluator

from swsearch.config import settings
from swsearch.logutil import get_logger
from swsearch.metadata.store import load_faiss_meta_sqlite

logger = get_logger(__name__)


def load_triplet_dataset(triplets_dir: str, holdout_files: int = 3, shuffle_buffer_size: int = 10_000):
    """Stream mined (anchor, positive, negative) triplets from JSONL files
    without loading the whole corpus into memory -- mining a full enwiki
    corpus can produce tens of millions of triplets, the same scale that
    OOM'd a flat FAISS index earlier this project. Extra columns the miner
    writes (source, url) are dropped; the loss/evaluator only want these
    three.

    The training stream is shuffled with a buffer -- mining/triplets.py
    writes every triplet for one article consecutively (they all share that
    article's lead paragraph as the anchor), so an unshuffled stream feeds
    the trainer long runs of same-anchor rows. That's harmless for
    TripletLoss (each row's loss is independent), but MultipleNegativesRankingLoss
    treats every other row's positive in a batch as an implicit negative for
    this row's anchor -- with same-anchor rows clustered together, most of
    those "negatives" are actually other valid positives for the same
    article, actively teaching the model a contradictory signal. A large
    shuffle buffer spreads same-article rows across many batches instead.

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

    train_dataset = (
        load_dataset("json", data_files=train_files, split="train", streaming=True)
        .select_columns(columns)
        .shuffle(buffer_size=shuffle_buffer_size)
    )
    eval_dataset = None
    if eval_files:
        eval_dataset = load_dataset("json", data_files=eval_files, split="train", streaming=False).select_columns(columns)

    return train_dataset, eval_dataset


def build_retrieval_evaluator(
    test_queries_path: str,
    meta_db_path: str,
    distractor_count: int = 500_000,
    max_paragraphs_per_article: int = 20,
    seed: int = 42,
    name: str = "wiki-ir",
) -> InformationRetrievalEvaluator | None:
    """Build an evaluator that runs the *real* retrieval task -- the eval
    queries from test_queries.json against a corpus of paragraphs -- so
    checkpoint selection optimises what search actually does.

    This exists because held-out triplet accuracy turned out to be an
    actively misleading model-selection signal. On a 500k-step run it climbed
    monotonically to 0.89 while real corpus MRR collapsed to 0.16: the mined
    triplets it scores are drawn from the same biased distribution the model
    is training on, so a model that learns that bias thoroughly scores well
    on them by construction. Ranking real queries against real paragraphs has
    no such shared failure mode.

    The corpus is a sample, not the full index: every paragraph from the
    articles the test set marks relevant (so the answers are present at all),
    plus distractor_count random paragraphs to keep the ranking non-trivial.
    Encoding all ~42M paragraphs at every checkpoint would cost more than the
    training run it is meant to guide (~3.4h per pass, and ~60GB held in RAM
    for the similarity search) -- and would still be measuring the same
    ordering this sample measures.

    distractor_count trades discrimination against eval time. Measured with
    the baseline model, 155 test queries, one eval pass:
        50k    MRR@10 0.9935   24s   -- saturated, cannot rank checkpoints
        500k   MRR@10 0.9753  184s   -- default
        2M     MRR@10 0.8830  774s   -- best separation, ~65m per 5-eval run
    The default is deliberately conservative: it is comfortably sensitive to
    the failure this exists to catch (a checkpoint whose retrieval has
    collapsed scores far below 0.97 here), while adding ~15m to a run rather
    than an hour. Raise it when comparing checkpoints that are all healthy
    and close together, where 0.975 vs 0.976 is noise.

    Absolute numbers run higher than `swsearch evaluate` against the full
    index either way; they are only meaningful for comparing checkpoints *to
    each other*, which is all load_best_model_at_end needs. Run the real
    evaluate afterwards for a number worth quoting.

    Returns None if the test set or metadata store is missing, so training
    can fall back to the triplet evaluator instead of failing outright.
    """
    if not os.path.exists(test_queries_path):
        logger.warning("Test queries not found at %s; cannot build retrieval evaluator", test_queries_path)
        return None
    if not os.path.exists(meta_db_path):
        logger.warning("FAISS metadata store not found at %s; cannot build retrieval evaluator", meta_db_path)
        return None

    with open(test_queries_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)

    conn = load_faiss_meta_sqlite(meta_db_path)
    try:
        cur = conn.cursor()
        return _build_evaluator_from_meta(
            cur, test_set, meta_db_path, distractor_count, max_paragraphs_per_article, seed, name
        )
    finally:
        conn.close()


def _build_evaluator_from_meta(cur, test_set, meta_db_path, distractor_count, max_paragraphs_per_article, seed, name):
    """Body of build_retrieval_evaluator, split out only so the connection
    it reads through is closed by a try/finally in the caller."""
    corpus: dict[str, str] = {}
    title_to_docs: dict[str, set[str]] = {}
    relevant_titles = {t for entry in test_set for t in entry.get("relevant_articles", [])}
    for title in relevant_titles:
        # Relies on faiss_meta's idx_article index -- without it this is a
        # full scan of a multi-GB table per title.
        cur.execute(
            "SELECT idx, text FROM faiss_meta WHERE article_title = ? LIMIT ?",
            (title, max_paragraphs_per_article),
        )
        docs = set()
        for row in cur.fetchall():
            doc_id = str(row["idx"])
            corpus[doc_id] = row["text"]
            docs.add(doc_id)
        if docs:
            title_to_docs[title] = docs

    cur.execute("SELECT MAX(idx) FROM faiss_meta")
    max_idx = cur.fetchone()[0]
    if max_idx:
        rng = random.Random(seed)
        # Sampling ids and fetching by primary key, rather than ORDER BY
        # RANDOM(), which would sort tens of millions of rows per call.
        wanted = {rng.randint(0, max_idx) for _ in range(distractor_count)} - set(map(int, corpus))
        for chunk_start in range(0, len(wanted), 5_000):
            chunk = list(wanted)[chunk_start:chunk_start + 5_000]
            placeholders = ",".join("?" * len(chunk))
            cur.execute(f"SELECT idx, text FROM faiss_meta WHERE idx IN ({placeholders})", chunk)
            for row in cur.fetchall():
                corpus.setdefault(str(row["idx"]), row["text"])

    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    for i, entry in enumerate(test_set):
        docs: set[str] = set()
        for title in entry.get("relevant_articles", []):
            docs |= title_to_docs.get(title, set())
        # A query whose relevant articles are all absent from the corpus is
        # unanswerable and would just drag every checkpoint down equally.
        if not docs:
            continue
        qid = str(i)
        queries[qid] = entry["query"]
        relevant_docs[qid] = docs

    if not queries:
        logger.warning("No test queries had relevant articles present in %s; falling back", meta_db_path)
        return None

    logger.info(
        "Retrieval evaluator: %d queries, %d corpus paragraphs (%d from relevant articles)",
        len(queries), len(corpus), sum(len(d) for d in title_to_docs.values()),
    )
    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=name,
        show_progress_bar=False,
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1, 5, 10],
        precision_recall_at_k=[10],
        map_at_k=[10],
    )
    # Select checkpoints on MRR@10 specifically -- it matches the headline
    # metric `swsearch evaluate` reports, so training-time selection and
    # after-the-fact reporting agree on what "better" means.
    evaluator.primary_metric = f"{name}_cosine_mrr@10"
    return evaluator


def _freeze_lower_layers(model: SentenceTransformer, freeze_layers: int) -> None:
    """Freeze the embeddings layer and the bottom freeze_layers transformer
    layers, leaving the top layers (and pooling) trainable. Lower layers of
    a pretrained transformer tend to encode general-purpose linguistic
    structure; upper layers are more task-specific. Everything observed
    fine-tuning this model points to catastrophic forgetting as the core
    tension -- a gentler run (fewer steps, lower learning rate) outperformed
    a more thorough one on otherwise-identical data -- so partial freezing
    protects the general-purpose structure the baseline's retrieval quality
    depends on while still letting the top of the network adapt.

    Assumes a BERT-style architecture (embeddings + encoder.layer, true for
    the default all-MiniLM-L6-v2); silently skipped with a warning for any
    base_model_name that doesn't expose that structure, rather than crashing
    the run over an optional adjustment.
    """
    if freeze_layers <= 0:
        return
    try:
        auto_model = model[0].auto_model
        for param in auto_model.embeddings.parameters():
            param.requires_grad = False
        for layer in auto_model.encoder.layer[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False
        logger.info("Froze embeddings + bottom %d transformer layer(s)", freeze_layers)
    except AttributeError:
        logger.warning("%s doesn't expose the expected embeddings/encoder.layer structure; skipping layer freezing", model[0].auto_model.__class__.__name__)


def train_transfer_model(
    triplets_dir: str,
    output_dir: str,
    base_model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    max_steps: int = 20000,
    learning_rate: float = 2e-5,
    scale: float = 20.0,
    warmup_ratio: float = 0.1,
    freeze_layers: int = 3,
    device: str | None = None,
    eval_meta_db: str | None = None,
    eval_distractors: int = 500_000,
) -> None:
    """Fine-tune base_model_name on triplets mined from the baseline index
    with MultipleNegativesRankingLoss, saving the best-scoring checkpoint
    (not necessarily the last step) to output_dir. Selection is by real
    retrieval MRR@10 when eval_meta_db is given, and by held-out triplet
    accuracy only as a fallback -- see build_retrieval_evaluator for why
    that fallback cannot be trusted on its own. That path can then be
    passed straight to `swsearch embed --model-name <output_dir>` --
    sentence-transformers loads a local folder the same way it loads a hub
    model name.

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
    picks whichever surviving checkpoint scores highest on the evaluator,
    not necessarily the final one -- protects against the model quietly
    getting worse over the course of a long run with nothing watching for
    it. Pass eval_meta_db to select on real retrieval (MRR@10 over the test
    queries against a sampled corpus -- see build_retrieval_evaluator);
    without it, selection falls back to held-out triplet accuracy, which a
    500k-step run showed rising to 0.89 while true corpus MRR fell to 0.16.
    Either way the sampled/held-out score is a stand-in for the full-index
    number: run `swsearch evaluate` after training for that.

    warmup_ratio ramps the learning rate up over the first fraction of
    steps instead of starting at full strength immediately -- standard
    practice when fine-tuning a pretrained transformer, so the first
    updates (potentially the most disruptive to already-good pretrained
    weights) are the gentlest ones. freeze_layers protects the bottom of
    the network from those updates entirely; see _freeze_lower_layers.
    """
    os.makedirs(output_dir, exist_ok=True)
    resolved_device = device or settings.model.device
    model = SentenceTransformer(base_model_name, device=resolved_device)
    _freeze_lower_layers(model, freeze_layers)
    train_dataset, eval_dataset = load_triplet_dataset(triplets_dir)
    loss = losses.MultipleNegativesRankingLoss(model=model, scale=scale)

    # Prefer selecting checkpoints on real retrieval, falling back to
    # held-out triplet accuracy only when no metadata store is available to
    # build a corpus from -- see build_retrieval_evaluator for why the
    # triplet metric can't be trusted on its own.
    evaluator = None
    # Set alongside each evaluator rather than derived from the evaluator
    # afterwards: TripletEvaluator.primary_metric is None until it has been
    # run at least once, so reading it here would name the metric "eval_None"
    # and silently disable best-checkpoint selection.
    metric_for_best_model = "eval_cosine_accuracy"
    if eval_meta_db:
        evaluator = build_retrieval_evaluator(
            test_queries_path=str(settings.paths.test_queries_path),
            meta_db_path=eval_meta_db,
            distractor_count=eval_distractors,
        )
        if evaluator is not None:
            metric_for_best_model = f"eval_{evaluator.primary_metric}"
    if evaluator is None and eval_dataset is not None:
        logger.warning("Falling back to TripletEvaluator; checkpoint selection will use held-out triplet accuracy")
        evaluator = TripletEvaluator(
            anchors=eval_dataset["anchor"],
            positives=eval_dataset["positive"],
            negatives=eval_dataset["negative"],
        )
        metric_for_best_model = "eval_cosine_accuracy"

    eval_steps = max(max_steps // 5, 1)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        fp16=(resolved_device == "cuda"),
        logging_steps=100,
        eval_strategy="steps" if evaluator is not None else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=5,
        load_best_model_at_end=evaluator is not None,
        metric_for_best_model=metric_for_best_model,
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
        logger.info("Best checkpoint: %s (%s=%s)", trainer.state.best_model_checkpoint, metric_for_best_model, trainer.state.best_metric)

    model.save(output_dir)
    logger.info("Fine-tuned model saved to %s", output_dir)
