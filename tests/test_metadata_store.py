from swsearch.metadata.store import (
    append_faiss_meta_batch,
    create_faiss_meta_db,
    get_text_and_meta_from_db,
    get_texts_and_meta_from_db,
    load_faiss_meta_sqlite,
)


def _seed(tmp_path, texts=("t0", "t1", "t2"), titles=("Article A", "Article A", "Article B")):
    db_path = str(tmp_path / "meta.db")
    write_conn = create_faiss_meta_db(db_path)
    append_faiss_meta_batch(write_conn, 0, list(texts), list(titles))
    write_conn.close()
    # Mining/search read through load_faiss_meta_sqlite, which sets
    # row_factory=sqlite3.Row -- get_text(s)_and_meta_from_db rely on that
    # for named column access, same as the real pipeline.
    return load_faiss_meta_sqlite(db_path)


def test_get_texts_and_meta_from_db_matches_single_lookup(tmp_path):
    conn = _seed(tmp_path)

    batched = get_texts_and_meta_from_db(conn, [0, 1, 2])

    assert batched == {
        0: ("t0", "Article A"),
        1: ("t1", "Article A"),
        2: ("t2", "Article B"),
    }
    for idx in (0, 1, 2):
        assert batched[idx] == get_text_and_meta_from_db(conn, idx)


def test_get_texts_and_meta_from_db_dedupes_and_ignores_missing_and_negative(tmp_path):
    conn = _seed(tmp_path)

    # Duplicate indices (as a real FAISS candidate list would have across
    # anchors), an out-of-range idx, and a -1 (FAISS's "no result") mixed in.
    batched = get_texts_and_meta_from_db(conn, [0, 0, 1, 1, 99, -1])

    assert batched == {0: ("t0", "Article A"), 1: ("t1", "Article A")}


def test_get_texts_and_meta_from_db_empty_input(tmp_path):
    conn = _seed(tmp_path)

    assert get_texts_and_meta_from_db(conn, []) == {}
    assert get_texts_and_meta_from_db(conn, [-1]) == {}


def test_get_texts_and_meta_from_db_chunks_large_index_lists(tmp_path):
    n = 1200  # more than the 500-per-query chunk size
    conn = _seed(tmp_path, texts=[f"t{i}" for i in range(n)], titles=[f"Article {i}" for i in range(n)])

    batched = get_texts_and_meta_from_db(conn, list(range(n)))

    assert len(batched) == n
    assert batched[0] == ("t0", "Article 0")
    assert batched[n - 1] == (f"t{n - 1}", f"Article {n - 1}")
