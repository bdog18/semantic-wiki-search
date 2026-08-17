import json
import sqlite3

import pytest

from swsearch.linkgraph.backlinks import build_backlink_counts_sqlite, load_backlink_counts_sqlite


def _seed_link_graph(tmp_path, rows: dict[str, list[str]]):
    """Fake from_title -> linked_titles link graph, same shape as
    linkgraph.store.build_linkgraph_sqlite produces."""
    db_path = str(tmp_path / "linkgraph.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE links (from_title TEXT PRIMARY KEY, linked_titles TEXT)")
    conn.executemany(
        "INSERT INTO links (from_title, linked_titles) VALUES (?, ?)",
        [(from_title, json.dumps(linked_titles)) for from_title, linked_titles in rows.items()],
    )
    conn.commit()
    conn.close()
    return db_path


def test_build_backlink_counts_sqlite_counts_incoming_links(tmp_path):
    link_db_path = _seed_link_graph(
        tmp_path,
        {
            "A": ["B", "C"],
            "B": ["C"],
            "D": ["C", "B"],
        },
    )
    out_db_path = str(tmp_path / "backlinks.db")

    skipped = build_backlink_counts_sqlite(link_db_path, out_db_path)

    assert skipped == 0
    conn = load_backlink_counts_sqlite(out_db_path)
    cur = conn.cursor()
    cur.execute("SELECT count FROM backlinks WHERE title = ?", ("B",))
    assert cur.fetchone()["count"] == 2  # linked from A and D
    cur.execute("SELECT count FROM backlinks WHERE title = ?", ("C",))
    assert cur.fetchone()["count"] == 3  # linked from A, B, and D


def test_build_backlink_counts_sqlite_skips_malformed_rows(tmp_path):
    db_path = str(tmp_path / "linkgraph.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE links (from_title TEXT PRIMARY KEY, linked_titles TEXT)")
    conn.executemany(
        "INSERT INTO links (from_title, linked_titles) VALUES (?, ?)",
        [
            ("A", json.dumps(["B"])),
            ("bad-row", "not valid json"),
        ],
    )
    conn.commit()
    conn.close()
    out_db_path = str(tmp_path / "backlinks.db")

    skipped = build_backlink_counts_sqlite(db_path, out_db_path)

    assert skipped == 1
    result_conn = load_backlink_counts_sqlite(out_db_path)
    cur = result_conn.cursor()
    cur.execute("SELECT count FROM backlinks WHERE title = ?", ("B",))
    assert cur.fetchone()["count"] == 1


def test_load_backlink_counts_sqlite_row_factory_supports_named_access(tmp_path):
    # heuristic.rerank reads backlink_row["count"] by column name -- this
    # only works if load_backlink_counts_sqlite sets row_factory=sqlite3.Row.
    link_db_path = _seed_link_graph(tmp_path, {"A": ["B"]})
    out_db_path = str(tmp_path / "backlinks.db")
    build_backlink_counts_sqlite(link_db_path, out_db_path)

    conn = load_backlink_counts_sqlite(out_db_path)

    assert isinstance(conn.row_factory, type) and conn.row_factory is sqlite3.Row


def test_load_backlink_counts_sqlite_missing_title_has_no_row(tmp_path):
    # A title that's never a link target doesn't get a row in `backlinks` at
    # all (the table is built from a Counter over observed link targets) --
    # callers (heuristic.rerank) must treat a missing row as count=0.
    link_db_path = _seed_link_graph(tmp_path, {"A": ["B"]})
    out_db_path = str(tmp_path / "backlinks.db")
    build_backlink_counts_sqlite(link_db_path, out_db_path)

    conn = load_backlink_counts_sqlite(out_db_path)
    cur = conn.cursor()
    cur.execute("SELECT count FROM backlinks WHERE title = ?", ("Never Linked",))

    assert cur.fetchone() is None


def test_load_backlink_counts_sqlite_raises_if_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.db")

    with pytest.raises(FileNotFoundError):
        load_backlink_counts_sqlite(missing_path)
