import numpy as np
import pytest

from swsearch.index.faiss_store import IndexAlignmentError, _default_ivfpq_params, build_flat_index_from_manifest
from swsearch.metadata.store import (
    append_faiss_meta_batch,
    create_faiss_meta_db,
    get_count_from_faiss_meta_db,
    record_manifest_entry,
)


def _write_batch(embeddings_dir, meta_conn, seq, start_idx, texts, titles, dim=4):
    vectors = np.random.rand(len(texts), dim).astype(np.float32)
    npy_filename = f"batch_{seq:05d}.npy"
    np.save(embeddings_dir / npy_filename, vectors)
    append_faiss_meta_batch(meta_conn, start_idx, texts, titles)
    record_manifest_entry(meta_conn, seq, npy_filename, len(texts))
    return len(texts)


def test_build_index_from_manifest_stays_aligned(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    meta_db_path = tmp_path / "meta.db"
    index_path = tmp_path / "flat.index"

    meta_conn = create_faiss_meta_db(str(meta_db_path))
    total = 0
    total += _write_batch(embeddings_dir, meta_conn, 0, total, ["a", "b", "c"], ["Article A"] * 3)
    total += _write_batch(embeddings_dir, meta_conn, 1, total, ["d", "e"], ["Article B"] * 2)

    index = build_flat_index_from_manifest(str(embeddings_dir), meta_conn, str(index_path))

    assert index.ntotal == total == get_count_from_faiss_meta_db(meta_conn)
    assert index_path.exists()


def test_build_index_from_manifest_raises_on_mismatch(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    meta_db_path = tmp_path / "meta.db"
    index_path = tmp_path / "flat.index"

    meta_conn = create_faiss_meta_db(str(meta_db_path))
    # Manifest claims 3 rows, but the .npy file actually written has only 2 --
    # exactly the drift bug this manifest design exists to catch loudly.
    vectors = np.random.rand(2, 4).astype(np.float32)
    np.save(embeddings_dir / "batch_00000.npy", vectors)
    append_faiss_meta_batch(meta_conn, 0, ["a", "b", "c"], ["Article A"] * 3)
    record_manifest_entry(meta_conn, 0, "batch_00000.npy", 3)

    with pytest.raises(IndexAlignmentError):
        build_flat_index_from_manifest(str(embeddings_dir), meta_conn, str(index_path))


def test_default_ivfpq_params_uses_floor_for_small_corpora():
    # A few-hundred-row smoke-test corpus shouldn't get an oversized nlist.
    nlist, train_size = _default_ivfpq_params(500)
    assert nlist == 100
    assert train_size == 500  # can't train on more points than exist


def test_default_ivfpq_params_scales_with_corpus_size():
    # ~70M paragraphs (a real full-enwiki run) should get a much larger nlist
    # than the 100 that was hardcoded before -- that's the whole point of
    # this: fixed nlist=100 left every cluster holding ~700k vectors.
    nlist, train_size = _default_ivfpq_params(70_365_524)
    assert nlist > 30_000
    assert nlist <= 65536  # capped
    assert train_size >= 40 * nlist  # stays above FAISS's training-point warning floor


def test_default_ivfpq_params_caps_nlist_for_huge_corpora():
    nlist, _ = _default_ivfpq_params(10_000_000_000)
    assert nlist == 65536
