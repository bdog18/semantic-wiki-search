import faiss
import numpy as np
import pytest

import swsearch.index.faiss_store as faiss_store
from swsearch.index.faiss_store import (
    IndexAlignmentError,
    _default_ivfpq_params,
    build_flat_index_from_manifest,
    build_ivfpq_index_from_manifest,
    load_index,
    resolve_index_type,
)
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


def test_reconstruct_works_after_loading_an_ivfpq_index(tmp_path):
    # IndexIVFPQ.reconstruct() raises RuntimeError without a direct map --
    # search.engine.SearchEngine relies on reconstruct() for its cosine-
    # similarity rerank, silently falling back to raw (uncalibrated) distance
    # if it fails. This is what a real ivfpq run hit live.
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    meta_db_path = tmp_path / "meta.db"
    index_path = tmp_path / "ivfpq.index"

    meta_conn = create_faiss_meta_db(str(meta_db_path))
    total = 0
    for seq in range(4):
        total += _write_batch(
            embeddings_dir, meta_conn, seq, total,
            [f"t{seq}-{i}" for i in range(50)], [f"Article {seq}"] * 50, dim=16,
        )

    build_ivfpq_index_from_manifest(
        str(embeddings_dir), meta_conn, str(index_path), nlist=4, m=4, nbits=4, train_size=total,
    )

    loaded = load_index(str(index_path))
    assert isinstance(loaded, faiss.IndexIVFPQ)
    vec = loaded.reconstruct(0)  # raises RuntimeError pre-fix
    assert vec.shape == (16,)


# --- index type resolution ---
# "auto" exists because the wrong choice is expensive in opposite directions:
# a flat index over the real corpus (41,953,396 x 384) is 60GB assembled in
# memory -- which is what OOM'd earlier in this project -- while IVF-PQ over a
# scratch corpus is lossy for nothing. The shipped index is IVF-PQ at 999MB.

def test_auto_picks_flat_for_a_scratch_corpus():
    assert resolve_index_type("auto", 5_518, 384) == "flat"


def test_auto_picks_ivfpq_for_the_real_corpus():
    assert resolve_index_type("auto", 41_953_396, 384) == "ivfpq"


def test_explicit_index_type_overrides_auto():
    assert resolve_index_type("flat", 41_953_396, 384) == "flat"
    assert resolve_index_type("ivfpq", 10, 384) == "ivfpq"


def test_unknown_index_type_is_rejected():
    with pytest.raises(ValueError):
        resolve_index_type("bogus", 10, 384)


def test_flat_builder_refuses_a_corpus_it_cannot_hold_in_memory(tmp_path, monkeypatch):
    """Fails on the manifest, before reading a single batch -- otherwise the
    error arrives partway through as an OOM that looks like a machine fault
    rather than a wrong flag."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    meta_db_path = tmp_path / "meta.db"
    meta_conn = create_faiss_meta_db(str(meta_db_path))
    # Real embedding width: the ceiling is a byte budget, so the dimension
    # matters as much as the row count.
    _write_batch(embeddings_dir, meta_conn, 0, 0, ["a", "b"], ["A", "B"], dim=384)

    # Claim a corpus-scale row count without writing corpus-scale data.
    monkeypatch.setattr(faiss_store, "get_manifest", lambda conn: [("batch_00000.npy", 41_953_396)])

    with pytest.raises(ValueError, match="Use --index-type ivfpq"):
        faiss_store.build_flat_index_from_manifest(str(embeddings_dir), meta_conn, str(tmp_path / "flat.index"))


def test_build_index_from_manifest_routes_small_corpora_to_flat(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    meta_db_path = tmp_path / "meta.db"
    meta_conn = create_faiss_meta_db(str(meta_db_path))
    _write_batch(embeddings_dir, meta_conn, 0, 0, ["a", "b", "c"], ["A", "A", "B"], dim=4)

    index = faiss_store.build_index_from_manifest(
        str(embeddings_dir), meta_conn, str(tmp_path / "auto.index"), index_type="auto"
    )

    assert isinstance(index, faiss.IndexFlat)
    assert index.ntotal == 3
