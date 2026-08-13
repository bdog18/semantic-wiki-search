import os

import numpy as np
from pyspark.sql.functions import col, explode, split, trim
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from swsearch.common.spark import get_spark_session
from swsearch.common.textsplit import PARAGRAPH_SPLIT_PATTERN
from swsearch.logutil import get_logger
from swsearch.metadata.store import append_faiss_meta_batch, create_faiss_meta_db, record_manifest_entry

logger = get_logger(__name__)


def embed_paragraphs(
    data_dir: str,
    embeddings_dir: str,
    meta_db_path: str,
    model_name: str,
    batch_size: int = 4096,
    encode_batch_size: int = 256,
    device: str = "cpu",
) -> int:
    """Split every article in data_dir into paragraphs, embed them in batches,
    and write embeddings + metadata together so they can never drift apart.

    Each batch is (1) encoded, (2) saved as one .npy file, (3) immediately
    recorded in the FAISS meta SQLite store at the exact idx range it will
    occupy, and (4) logged in the manifest table in that same order.
    index/faiss_store.py later builds the FAISS index by replaying the
    manifest, not by re-deriving order from sorted(glob(...)) -- the two can
    no longer silently drift apart, and this also avoids holding the entire
    corpus's texts/embeddings in memory at once.

    batch_size and encode_batch_size are two different things despite the
    similar names: batch_size is how many paragraphs accumulate before one
    encode/save/write cycle runs at all; encode_batch_size is how many of
    those get encoded together in one actual GPU forward pass inside that
    cycle. Before this parameter existed, model.encode() was called with no
    batch_size of its own, so it silently used sentence-transformers'
    library default (32) regardless of how large batch_size was set to --
    e.g. a batch_size=1024 run was still only ever doing 32-at-a-time GPU
    calls, 32x smaller than intended. This runs in a single process (unlike
    mining/triplets.py's multi-worker encode() calls), so there's no
    multiply-by-worker-count memory concern here -- 256 is the same value
    already proven safe under mining's harder case (several concurrent
    worker processes each doing their own encode() calls).

    batch_size defaults much larger than encode_batch_size for a reason
    that isn't obvious: encode() sorts its *entire input list* by length
    before slicing it into encode_batch_size-sized GPU batches (grouping
    similar-length texts to minimize padding waste). That sort operates
    over whatever list it's handed -- i.e. batch_size's worth, not the
    whole corpus. A small batch_size means each GPU batch is a wide, poorly
    length-matched slice of a small sorted pool (e.g. one 256-batch was 25%
    of a 1024-item pool -- lots of length variance, lots of padding
    waste); a large batch_size gives encode() a much bigger pool to sort
    over, so the same 256-item GPU batch becomes a narrow, length-homogeneous
    slice instead (256/4096 = ~6%). Confirmed live this was actually
    regressing throughput at batch_size=1024: GPU utilization sat at 43%,
    consistent with cycles going to padding rather than useful compute.

    Returns the total number of paragraphs embedded.
    """
    os.makedirs(embeddings_dir, exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    meta_conn = create_faiss_meta_db(meta_db_path)

    spark = get_spark_session("swsearch-embed")
    try:
        logger.info("Loading JSON into Spark DataFrame from %s...", data_dir)
        df = spark.read.option("multiLine", True).option("recursiveFileLookup", "true").json(data_dir)

        df = df.withColumn("paragraphs", split(col("content"), PARAGRAPH_SPLIT_PATTERN))
        df = df.select("title", explode("paragraphs").alias("paragraph"))
        df = df.filter(trim(col("paragraph")) != "")

        total = 0
        batch_number = 0
        batch_texts: list[str] = []
        batch_titles: list[str] = []

        def flush_batch() -> None:
            nonlocal total, batch_number
            if not batch_texts:
                return
            embeddings = model.encode(
                batch_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=encode_batch_size
            ).astype(np.float32)

            npy_filename = f"batch_{batch_number:05d}.npy"
            np.save(os.path.join(embeddings_dir, npy_filename), embeddings)

            append_faiss_meta_batch(meta_conn, total, batch_texts, batch_titles)
            record_manifest_entry(meta_conn, batch_number, npy_filename, len(batch_texts))

            total += len(batch_texts)
            batch_number += 1
            batch_texts.clear()
            batch_titles.clear()

        logger.info("Encoding paragraphs in batches of %d...", batch_size)
        for row in tqdm(df.toLocalIterator(), desc="Encoding paragraphs", unit="paragraph"):
            batch_texts.append(row["paragraph"])
            batch_titles.append(row["title"])
            if len(batch_texts) >= batch_size:
                flush_batch()

        flush_batch()
    finally:
        spark.stop()
        meta_conn.close()

    logger.info("Done embedding %d paragraphs into %s", total, embeddings_dir)
    return total
