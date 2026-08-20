import os
import sqlite3


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faiss_meta (
            idx INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            article_title TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_article ON faiss_meta(article_title)")
    # Records the exact (npy_filename, row_count) order rows were written in,
    # so index/faiss_store.py can rebuild the FAISS index by loading files in
    # that same order instead of sorted(glob(...)), which has no guaranteed
    # correspondence to how idx values were assigned here.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS manifest (
            seq INTEGER PRIMARY KEY,
            npy_filename TEXT NOT NULL,
            row_count INTEGER NOT NULL
        )
    """)
    conn.commit()


def create_faiss_meta_db(db_path: str) -> sqlite3.Connection:
    """Create a fresh FAISS metadata database (removing any existing one)."""
    if os.path.exists(db_path):
        os.remove(db_path)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    _create_schema(conn)
    return conn


def append_faiss_meta_batch(conn: sqlite3.Connection, start_idx: int, texts: list[str], article_titles: list[str]) -> None:
    """Insert one encoded batch's metadata rows, keyed by the FAISS index
    position they will occupy (start_idx .. start_idx + len(texts) - 1)."""
    rows = [(start_idx + i, text, title) for i, (text, title) in enumerate(zip(texts, article_titles))]
    conn.executemany("INSERT INTO faiss_meta VALUES (?, ?, ?)", rows)
    conn.commit()


def record_manifest_entry(conn: sqlite3.Connection, seq: int, npy_filename: str, row_count: int) -> None:
    conn.execute("INSERT INTO manifest VALUES (?, ?, ?)", (seq, npy_filename, row_count))
    conn.commit()


def get_manifest(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """Return [(npy_filename, row_count), ...] in the exact order embeddings
    were written, i.e. the order the FAISS index must be built in to stay
    aligned with the idx values already recorded in faiss_meta."""
    cur = conn.cursor()
    cur.execute("SELECT npy_filename, row_count FROM manifest ORDER BY seq")
    return [(row[0], row[1]) for row in cur.fetchall()]


def load_faiss_meta_sqlite(db_path: str) -> sqlite3.Connection:
    """Load and return connection to FAISS metadata database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"FAISS metadata database not found at {db_path}")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_text_and_meta_from_db(conn: sqlite3.Connection, idx: int) -> tuple[str | None, str | None]:
    """Retrieve (text, article_title) by FAISS index position, or (None, None)."""
    cur = conn.cursor()
    cur.execute("SELECT text, article_title FROM faiss_meta WHERE idx = ?", (int(idx),))
    result = cur.fetchone()
    if result:
        return result["text"], result["article_title"]
    return None, None


def get_texts_and_meta_from_db(conn: sqlite3.Connection, indices: list[int]) -> dict[int, tuple[str, str]]:
    """Batched counterpart to get_text_and_meta_from_db: one query (chunked
    to stay under SQLite's bound-parameter limit) instead of one round trip
    per idx. Built for triplet mining's negative-candidate lookups, where a
    single mining batch can otherwise issue thousands of individual queries.
    Returns {idx: (text, article_title)} for whichever indices exist;
    missing/negative indices are simply absent from the result.
    """
    unique_indices = list({int(i) for i in indices if i >= 0})
    if not unique_indices:
        return {}

    result: dict[int, tuple[str, str]] = {}
    chunk_size = 500
    cur = conn.cursor()
    for start in range(0, len(unique_indices), chunk_size):
        chunk = unique_indices[start : start + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        cur.execute(f"SELECT idx, text, article_title FROM faiss_meta WHERE idx IN ({placeholders})", chunk)
        for row in cur.fetchall():
            result[row["idx"]] = (row["text"], row["article_title"])
    return result


def get_count_from_faiss_meta_db(conn: sqlite3.Connection) -> int:
    """Get total count of entries in FAISS metadata database."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM faiss_meta")
    return cur.fetchone()[0]
