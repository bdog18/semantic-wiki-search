import json
import os
import sqlite3

from tqdm import tqdm
from collections import Counter

from swsearch.logutil import get_logger

logger = get_logger(__name__)


def build_backlink_counts_sqlite(link_db_path: str, out_db_path: str) -> int:
    """Build the backlink counts SQLite lookup from link-graph SQLite."""
    conn = sqlite3.connect(link_db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM links")
    total_rows = cur.fetchone()[0]
    
    cur.execute("SELECT from_title, linked_titles FROM links")
    skipped = 0
    ctr = Counter()
    for from_title, linked_titles_json in tqdm(cur, total=total_rows, desc="Building backlink counts"):
        try:
            linked_titles = json.loads(linked_titles_json)
            for title in linked_titles:
                ctr[title] += 1
        except Exception:
            skipped += 1
            logger.exception("Skipping malformed link-graph row")
    conn.close()
    
    conn = sqlite3.connect(out_db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS backlinks")
    cur.execute("CREATE TABLE backlinks (title TEXT PRIMARY KEY, count INTEGER)")
    for title, count in tqdm(ctr.items(), desc="Inserting backlink counts"):
        cur.execute(
            "INSERT INTO backlinks (title, count) VALUES (?, ?)",
            (title, count)
        )
    conn.commit()
    conn.close()
    
    if skipped:
        logger.warning("Skipped %d malformed link-graph row(s) while building %s", skipped, out_db_path)
    return skipped


def load_backlink_counts_sqlite(db_path: str) -> sqlite3.Connection:
    """Load and return connection to backlinks count database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"backlinks count database not found at {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn