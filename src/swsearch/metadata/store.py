import os
import sqlite3

from tqdm import tqdm

from swsearch.logutil import get_logger

logger = get_logger(__name__)


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
    conn = sqlite3.connect(db_path)
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


def get_count_from_faiss_meta_db(conn: sqlite3.Connection) -> int:
    """Get total count of entries in FAISS metadata database."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM faiss_meta")
    return cur.fetchone()[0]


def build_faiss_meta_sqlite(all_texts: list[str], text_to_meta: list[str], db_path: str) -> str:
    """One-shot bulk builder for FAISS metadata, kept for the legacy
    meta.json -> SQLite conversion path (convert_faiss_meta_json_to_sqlite).
    The live embedding pipeline (embed/paragraphs.py) writes incrementally via
    append_faiss_meta_batch instead, since it never holds all_texts in memory.
    """
    logger.info("Creating FAISS metadata database at %s (%d texts)...", db_path, len(all_texts))

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    _create_schema(conn)
    cursor = conn.cursor()

    batch_size = 10000
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(all_texts), batch_size), total=total_batches, desc="Building FAISS meta DB"):
        batch_end = min(batch_start + batch_size, len(all_texts))
        batch = [(j, all_texts[j], text_to_meta[j]) for j in range(batch_start, batch_end)]
        cursor.executemany('INSERT INTO faiss_meta VALUES (?, ?, ?)', batch)
        conn.commit()

    count = get_count_from_faiss_meta_db(conn)
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    logger.info("Database created with %d entries (%.2f MB)", count, db_size_mb)

    conn.close()
    return db_path


def convert_faiss_meta_json_to_sqlite(json_path: str, db_path: str) -> None:
    """One-time conversion utility: existing paragraphs.index.meta.json -> SQLite."""
    import json

    logger.info("Converting %s to SQLite database...", json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    all_texts = meta['all_texts']
    text_to_meta = meta['text_to_meta']
    logger.info("Loaded %d texts", len(all_texts))

    build_faiss_meta_sqlite(all_texts, text_to_meta, db_path)
    logger.info("Conversion complete: %s -> %s", json_path, db_path)
