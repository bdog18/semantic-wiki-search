import json
import os
import sqlite3

from tqdm import tqdm

from swsearch.logutil import get_logger

logger = get_logger(__name__)


def build_linkgraph_sqlite(jsonl_dir: str, db_path: str) -> int:
    """Build the from_title -> linked_titles SQLite lookup from link-graph JSONL.

    Returns the number of rows skipped due to malformed input, logging each
    failure instead of silently dropping it.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS links (from_title TEXT PRIMARY KEY, linked_titles TEXT)")

    skipped = 0
    for root, _, files in os.walk(jsonl_dir):
        for file in tqdm(files, desc="Building SQLite"):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        cur.execute(
                            "INSERT INTO links (from_title, linked_titles) VALUES (?, ?)",
                            (entry["from_title"], json.dumps(entry["linked_titles"])),
                        )
                    except Exception:
                        skipped += 1
                        logger.exception("Skipping malformed link-graph row in %s", file)

    conn.commit()
    conn.close()
    if skipped:
        logger.warning("Skipped %d malformed link-graph row(s) while building %s", skipped, db_path)
    return skipped


def load_linkgraph_sqlite(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_links_for_title_sqlite(conn: sqlite3.Connection, title: str) -> set[str]:
    cur = conn.cursor()
    cur.execute("SELECT linked_titles FROM links WHERE from_title = ?", (title,))
    row = cur.fetchone()
    if row:
        # Titles are unescaped once, at write time, by linkgraph.build.unescape_mysql_string.
        return set(json.loads(row["linked_titles"]))
    return set()
