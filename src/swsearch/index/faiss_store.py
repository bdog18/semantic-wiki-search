import os

import faiss
import numpy as np
from tqdm import tqdm

from swsearch.logutil import get_logger
from swsearch.metadata.store import get_count_from_faiss_meta_db, get_manifest

logger = get_logger(__name__)


class IndexAlignmentError(RuntimeError):
    """Raised when a freshly built FAISS index's vector count doesn't match
    the row count already committed to the FAISS metadata SQLite store."""


def load_index(index_path: str) -> faiss.Index:
    logger.info("Loading FAISS index from %s", index_path)
    return faiss.read_index(index_path)


def query_faiss(index_path: str, query_embedding: np.ndarray, k: int) -> np.ndarray:
    index = load_index(index_path)
    _, indices = index.search(query_embedding, k)
    return indices


def _iter_manifest_batches(embeddings_dir: str, manifest: list[tuple[str, int]]):
    for npy_filename, expected_rows in manifest:
        path = os.path.join(embeddings_dir, npy_filename)
        vectors = np.load(path, mmap_mode='r')
        if len(vectors) != expected_rows:
            raise IndexAlignmentError(
                f"{npy_filename}: manifest says {expected_rows} rows, file has {len(vectors)}"
            )
        yield vectors


def _assert_aligned(index: faiss.Index, meta_conn) -> None:
    expected = get_count_from_faiss_meta_db(meta_conn)
    if index.ntotal != expected:
        raise IndexAlignmentError(
            f"Built index has {index.ntotal} vectors but FAISS meta DB has {expected} rows "
            "-- index and metadata are misaligned."
        )


def build_flat_index_from_manifest(embeddings_dir: str, meta_conn, index_path: str, batch_size: int = 1000) -> faiss.Index:
    """Build a flat IndexFlatL2 index by loading .npy embedding batches in the
    exact order recorded in the metadata manifest -- not sorted(glob(...)),
    which had no guaranteed correspondence to how metadata idx values were
    assigned during embedding. Fails loudly if the result doesn't line up
    with the metadata row count, rather than silently drifting.
    """
    manifest = get_manifest(meta_conn)
    if not manifest:
        raise ValueError(f"No manifest entries found for {embeddings_dir}; was `swsearch embed` run against this meta db?")

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

    index = None
    logger.info("Building flat FAISS index from %d manifest entries...", len(manifest))
    for vectors in tqdm(_iter_manifest_batches(embeddings_dir, manifest), total=len(manifest), desc="Adding to FAISS index"):
        if index is None:
            index = faiss.IndexFlatL2(vectors.shape[1])
        for i in range(0, len(vectors), batch_size):
            index.add(vectors[i:i + batch_size].astype(np.float32))

    _assert_aligned(index, meta_conn)
    faiss.write_index(index, index_path)
    logger.info("FAISS index saved to %s (%d vectors, verified aligned with meta DB)", index_path, index.ntotal)
    return index


def build_ivfpq_index_from_manifest(
    embeddings_dir: str,
    meta_conn,
    index_path: str,
    nlist: int = 100,
    nprobe: int = 10,
    train_size: int = 100000,
    batch_size: int = 1000,
    m: int = 16,
    nbits: int = 8,
) -> faiss.Index:
    manifest = get_manifest(meta_conn)
    if not manifest:
        raise ValueError(f"No manifest entries found for {embeddings_dir}; was `swsearch embed` run against this meta db?")

    logger.info("Sampling vectors for IVF-PQ training...")
    sampled = []
    for vectors in _iter_manifest_batches(embeddings_dir, manifest):
        for vec in vectors:
            sampled.append(vec.astype(np.float32))
            if len(sampled) >= train_size:
                break
        if len(sampled) >= train_size:
            break
    train_data = np.vstack(sampled)

    dim = train_data.shape[1]
    if dim % m != 0:
        raise ValueError(f"Dimension {dim} must be divisible by m={m} for PQ indexing.")

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index.train(train_data)
    index.nprobe = nprobe

    logger.info("Adding embeddings to IVF-PQ index from %d manifest entries...", len(manifest))
    for vectors in tqdm(_iter_manifest_batches(embeddings_dir, manifest), total=len(manifest), desc="Adding to IVF-PQ index"):
        for i in range(0, len(vectors), batch_size):
            index.add(vectors[i:i + batch_size].astype(np.float32))

    _assert_aligned(index, meta_conn)
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)
    logger.info("IVF-PQ FAISS index saved to %s (%d vectors)", index_path, index.ntotal)
    return index
