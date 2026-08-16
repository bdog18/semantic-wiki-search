import gc
import json
import os
import random
import re
import time
from multiprocessing import cpu_count, get_context

import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from swsearch.common.textsplit import split_paragraphs
from swsearch.config import settings
from swsearch.linkgraph.store import get_links_for_title_sqlite, load_linkgraph_sqlite
from swsearch.logutil import get_logger
from swsearch.metadata.store import get_texts_and_meta_from_db, load_faiss_meta_sqlite

logger = get_logger(__name__)

_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s")


def _first_sentence(text: str, max_len: int = 200) -> str:
    """A lightweight (non-NLP) approximation of "the first sentence" --
    real sentence tokenization isn't worth the dependency/cost at this
    corpus's scale, and this only needs to be roughly right, not exact."""
    match = _SENTENCE_END_RE.search(text[:max_len])
    if match:
        return text[: match.start() + 1].strip()
    return text[:max_len].strip()

# Globals populated once per worker process by init_worker (workers are
# spawned -- see get_context("spawn") below -- so these are set up once per
# process and reused across every file that process handles, not
# re-initialized per task).
_model = None
_index = None
_faiss_meta_conn = None
_article_titles = None
_link_conn = None


def init_worker(faiss_index_path: str, faiss_meta_db_path: str, article_titles_path: str, link_graph_path: str, model_name: str) -> None:
    global _model, _index, _faiss_meta_conn, _article_titles, _link_conn

    worker_id = os.getpid()
    start_time = time.time()
    logger.info("Worker %d: initializing...", worker_id)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cpu_index = faiss.read_index(faiss_index_path)

    if settings.model.use_gpu_faiss and torch.cuda.is_available():
        logger.info("Worker %d: transferring FAISS index to GPU with FP16 (use_gpu_faiss=True)...", worker_id)
        res = faiss.StandardGpuResources()
        res.setTempMemory(2 * 1024 * 1024 * 1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        co.usePrecomputed = False
        _index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    else:
        _index = cpu_index

    _faiss_meta_conn = load_faiss_meta_sqlite(faiss_meta_db_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _model = SentenceTransformer(model_name, device=device)
    if device == 'cuda':
        _model.half()

    _link_conn = load_linkgraph_sqlite(link_graph_path)
    with open(article_titles_path, "r", encoding="utf-8") as f:
        _article_titles = json.load(f)

    logger.info("Worker %d: ready in %.1fs", worker_id, time.time() - start_time)


def process_batch(anchors: list[str], positives: list[str], metadata: list[tuple[str, set[str]]], out_f, neg_pool_size: int) -> int:
    """Encode a batch of anchor/positive pairs and mine one hard negative each
    from the FAISS index, excluding paragraphs from the same or a linked
    article. The negative is sampled randomly from the filtered candidate
    pool rather than always taking the nearest neighbor -- always picking
    the single hardest candidate has no relief once the model separates it
    from the anchor, and the link graph is an incomplete proxy for "these
    are actually unrelated" (two topically-close articles aren't always
    mutually hyperlinked), so the nearest hit is disproportionately likely
    to be a false negative. Sampling across the pool still yields hard
    negatives without letting any single systematic false negative dominate
    training. Returns the number of triplets written."""
    if not anchors or not _model or not _index or not _faiss_meta_conn:
        return 0

    min_len = settings.mining.min_paragraph_length
    triplets_count = 0
    try:
        anchor_embeddings = _model.encode(
            anchors,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=settings.mining.batch_size,
            show_progress_bar=False,
        )

        _, I = _index.search(anchor_embeddings.astype('float32'), neg_pool_size)

        # One batched SQLite lookup for every candidate in the whole mining
        # batch instead of one round trip per candidate -- with neg_pool_size
        # candidates checked per anchor (no early break, since negatives are
        # sampled from the full pool), that was up to
        # len(anchors) * neg_pool_size individual queries per batch.
        candidate_lookup = get_texts_and_meta_from_db(_faiss_meta_conn, [int(j) for row in I for j in row])

        for idx, (anchor, positive, (src_title, linked_titles)) in enumerate(zip(anchors, positives, metadata)):
            candidates = []
            for j in I[idx]:
                neg_para, neg_title = candidate_lookup.get(int(j), (None, None))
                if neg_para and neg_title and neg_title != src_title and neg_title not in linked_titles:
                    if len(neg_para) > min_len and neg_para != anchor and neg_para != positive:
                        candidates.append(neg_para)
            negative = random.choice(candidates) if candidates else None

            if negative:
                triplet = {
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative,
                    "source": src_title,
                    "url": (_article_titles or {}).get(src_title, ""),
                }
                out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                triplets_count += 1

    except Exception:
        logger.exception("Batch encode/search failed; dropping %d anchors", len(anchors))
        return 0

    return triplets_count


def process_file_worker(args: tuple[str, str, int]) -> tuple[str, int]:
    """Mine triplets from one JSONL file of articles. Renames the file to
    `<name>.completed` on success so a later run can resume without redoing
    work."""
    file_path, triplet_output_dir, file_index = args
    m = settings.mining

    output_path = os.path.join(triplet_output_dir, f"wiki_{file_index}_triplets.jsonl")
    triplets_written = 0
    skipped_lines = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f, open(output_path, "w", encoding="utf-8") as out_f:
            batch_anchors: list[str] = []
            batch_positives: list[str] = []
            batch_metadata: list[tuple[str, set[str]]] = []

            for line in f:
                try:
                    article = json.loads(line)
                except json.JSONDecodeError:
                    skipped_lines += 1
                    logger.exception("Skipping malformed JSONL line in %s", file_path)
                    continue

                title = article.get("title")
                if not title:
                    continue

                raw_text = article.get("content", "")
                if len(raw_text) < m.min_text_length:
                    continue

                # min_positive_words on top of the character-length filter:
                # a 30-char string can be "See also: History of science."
                # -- not a substantive paragraph, just short prose that
                # happens to clear the character floor. Word count is a
                # cheap, dependency-free proxy for "this is actually a
                # sentence/paragraph," not boilerplate.
                paras = [
                    p for p in split_paragraphs(raw_text)
                    if len(p) > m.min_paragraph_length and len(p.split()) >= m.min_positive_words
                ]
                if len(paras) < m.min_paragraphs_for_triplets:
                    continue

                linked_titles = get_links_for_title_sqlite(_link_conn, title)
                if not linked_titles:
                    continue

                # Anchor is usually the article title -- short, names a
                # topic rather than describing it in a sentence, and
                # structurally much closer to a search query than any
                # paragraph is (confirmed empirically: title anchors beat
                # both lead-paragraph and full-question-template anchors on
                # real eval). A minority of articles instead anchor on the
                # first sentence of the lead paragraph: real, naturally-
                # occurring prose (not a synthetic template, which diluted
                # signal and measurably hurt), giving the training data a
                # little sentence-shaped variety without repeating the
                # template mistake. Chosen once per article, not per
                # triplet, same as the title-only anchor was.
                if random.random() < m.sentence_anchor_probability:
                    anchor = _first_sentence(paras[0])
                else:
                    anchor = title
                for positive in paras[: m.max_triplets_per_article]:
                    batch_anchors.append(anchor)
                    batch_positives.append(positive)
                    batch_metadata.append((title, linked_titles))

                if len(batch_anchors) >= m.batch_size:
                    triplets_written += process_batch(batch_anchors, batch_positives, batch_metadata, out_f, m.negative_pool_size)
                    batch_anchors.clear()
                    batch_positives.clear()
                    batch_metadata.clear()
                    if triplets_written % 1000 == 0:
                        gc.collect()

            if batch_anchors:
                triplets_written += process_batch(batch_anchors, batch_positives, batch_metadata, out_f, m.negative_pool_size)

        if skipped_lines:
            logger.warning("%s: skipped %d malformed line(s)", file_path, skipped_lines)

        _rename_completed_file(file_path)
        return file_path, triplets_written

    except Exception:
        logger.exception("Failed to process %s", file_path)
        return file_path, 0


def mine_triplets(
    jsonl_dir: str,
    index_path: str,
    meta_db_path: str,
    link_db_path: str,
    article_titles_path: str,
    out_dir: str,
    num_workers: int | None = None,
    max_files_per_worker: int | None = None,
) -> int:
    """Memory-efficient parallel triplet mining over JSONL articles, using an
    existing FAISS paragraph index + metadata store to find hard negatives."""
    os.makedirs(out_dir, exist_ok=True)

    empty_count = 0
    for filename in os.listdir(out_dir):
        if filename.endswith("_triplets.jsonl"):
            filepath = os.path.join(out_dir, filename)
            if os.path.getsize(filepath) == 0:
                os.remove(filepath)
                empty_count += 1
    if empty_count:
        logger.info("Removed %d empty triplet file(s) from a previous interrupted run", empty_count)

    all_files = sorted(
        os.path.join(root, file)
        for root, _, filenames in os.walk(jsonl_dir)
        for file in filenames
        if file.endswith(".jsonl") or file.endswith(".jsonl.completed")
    )

    completed = {f[: -len(".completed")] if f.endswith(".completed") else f for f in all_files if f.endswith(".completed")}
    files_to_process = [f for f in all_files if not f.endswith(".completed") and f not in completed]

    if not files_to_process:
        logger.info("No .jsonl files to process (all files already completed).")
        return 0

    logger.info("Found %d total files, %d already completed, %d to process", len(all_files), len(completed), len(files_to_process))

    existing = [f for f in os.listdir(out_dir) if f.startswith("wiki_") and f.endswith("_triplets.jsonl")]
    max_existing_index = 0
    for filename in existing:
        try:
            index = int(filename.split("_")[1])
            max_existing_index = max(max_existing_index, index)
        except (ValueError, IndexError):
            continue
    start_index = max_existing_index + 1

    if num_workers is None:
        num_workers = min(cpu_count() // 2, 2) or 1
    if max_files_per_worker is None:
        max_files_per_worker = max(200, len(files_to_process) // num_workers)

    logger.info("Mining triplets with %d workers over %d files...", num_workers, len(files_to_process))
    start_time = time.time()

    # Explicit "spawn" context, not the default Pool(): each worker loads its
    # own SentenceTransformer onto CUDA (init_worker), but the parent process
    # already touches CUDA itself (swsearch.config's device auto-detection
    # calls torch.cuda.is_available() at import time). Forking from a
    # CUDA-touched parent -- which is what plain Pool() does here, since this
    # Python's default start method is "forkserver", not "fork" -- inherits
    # that already-initialized CUDA state, and every worker's own CUDA init
    # then fails with "Cannot re-initialize CUDA in forked subprocess". Spawn
    # starts each worker as a fresh process with no inherited CUDA state.
    with get_context("spawn").Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(index_path, meta_db_path, article_titles_path, link_db_path, settings.model.embedding_model_name),
        maxtasksperchild=max_files_per_worker,
    ) as pool:
        worker_args = [(file_path, out_dir, start_index + idx) for idx, file_path in enumerate(files_to_process)]
        results = list(tqdm(
            pool.imap_unordered(process_file_worker, worker_args),
            total=len(files_to_process),
            desc="Mining triplets",
            unit="file",
        ))

    total_triplets = sum(n for _, n in results)
    elapsed = time.time() - start_time
    logger.info("Mined %d triplets from %d files in %.1f min", total_triplets, len(results), elapsed / 60)
    return total_triplets


def _rename_completed_file(src_path: str) -> None:
    completed_path = src_path + ".completed"
    os.replace(src_path, completed_path)
