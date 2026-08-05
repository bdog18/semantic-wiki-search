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
    batch_size: int = 1024,
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
                batch_texts, convert_to_numpy=True, normalize_embeddings=True
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
