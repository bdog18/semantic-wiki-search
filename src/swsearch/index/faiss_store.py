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


def load_index(index_path: str, mmap: bool = False) -> faiss.Index:
    """Load a FAISS index, building a direct map for IVF-family indexes
    (IVF-PQ, IVF-Flat, ...) so `.reconstruct()` works on them.

    mmap=True maps the file instead of reading it into the heap. Peak usage
    is barely different (measured 1.77GB vs 1.95GB), so this is not a way to
    use less memory -- it is a way to use *reclaimable* memory. Mapped file
    pages are clean and the kernel can drop them under pressure; heap
    allocations cannot be dropped, so a constrained sandbox OOM-kills
    instead. That distinction is what matters on a platform that counts
    file-backed pages against a hard ceiling, such as Lambda.

    Search results are unaffected: ids, distances and reconstruct() were
    verified byte-identical between the two paths.

    Without this, IndexIVFPQ.reconstruct() raises RuntimeError, which
    search.engine.SearchEngine silently catches and falls back to raw (PQ-
    approximated) L2 distance instead of the intended cosine-similarity
    rerank -- functional, but the resulting "scores" are an uncalibrated,
    unbounded distance rather than the expected -1..1 cosine range, and
    ranking quality suffers since it's comparing lossy PQ distances instead
    of reconstructed-vector cosine similarity.
    """
    logger.info("Loading FAISS index from %s (mmap=%s)", index_path, mmap)
    index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP) if mmap else faiss.read_index(index_path)
    if isinstance(index, faiss.IndexIVF):
        index.make_direct_map()
        # Avoid IVF's default parallel_mode=0 ("parallelise over queries"),
        # which on this build (faiss-cpu 1.15.0, generic -- it logs that it
        # could load neither the AVX2 nor the AVX512 library) returns
        # corrupted results for multi-query searches against a corpus-scale
        # index: searching 4 vectors at once returned one id repeated at
        # distance -0.019 where the true nearest neighbours are at 0.417,
        # while the same queries issued one at a time were correct.
        # Single-query callers (search.engine.SearchEngine) never hit it,
        # but nothing should have to know that to be safe. parallel_mode=3
        # parallelises over inverted lists and was verified to reproduce
        # single-threaded output exactly.
        index.parallel_mode = 3
    return index


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


def _default_ivfpq_params(total_rows: int) -> tuple[int, int]:
    """nlist/train_size scaled to corpus size.

    FAISS's own rule of thumb is nlist ~ 4*sqrt(N) to 16*sqrt(N); a fixed
    nlist=100 (fine for a few-hundred-row smoke test) means every cluster
    holds a huge fraction of the corpus at real Wikipedia scale (~70M
    paragraphs), making every query scan a large chunk of the index. Scale it
    with corpus size instead, capped so training and per-query
    cluster-selection overhead stay reasonable. train_size is kept above
    FAISS's ~39-training-points-per-centroid warning threshold rather than
    left at a fixed 100k that may under-train a larger nlist.
    """
    nlist = max(100, min(65536, int(4 * (total_rows ** 0.5))))
    train_size = min(total_rows, max(100_000, 40 * nlist))
    return nlist, train_size


def build_ivfpq_index_from_manifest(
    embeddings_dir: str,
    meta_conn,
    index_path: str,
    nlist: int | None = None,
    nprobe: int = 10,
    train_size: int | None = None,
    batch_size: int = 1000,
    m: int = 16,
    nbits: int = 8,
) -> faiss.Index:
    manifest = get_manifest(meta_conn)
    if not manifest:
        raise ValueError(f"No manifest entries found for {embeddings_dir}; was `swsearch embed` run against this meta db?")

    total_rows = sum(row_count for _, row_count in manifest)
    default_nlist, default_train_size = _default_ivfpq_params(total_rows)
    if nlist is None:
        nlist = default_nlist
    if train_size is None:
        train_size = default_train_size

    logger.info("Sampling vectors for IVF-PQ training (nlist=%d, train_size=%d)...", nlist, train_size)
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
