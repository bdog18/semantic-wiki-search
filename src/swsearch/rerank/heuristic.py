import math
import sqlite3

# SQLite's default bound-parameter ceiling is 999; 500 matches the chunk size
# metadata.store.get_texts_and_meta_from_db already uses for the same reason.
_LOOKUP_CHUNK_SIZE = 500


def _backlink_counts(conn: sqlite3.Connection, titles: list[str]) -> dict[str, int]:
    """Fetch backlink counts for every candidate title in one query.

    This used to be a cursor and a SELECT per candidate inside the scoring
    loop -- up to 50 round trips against a 1.3GB table for a single search.
    Same shape metadata.store and metadata.backends already fixed; see
    metadata/backends.py's module docstring for why batch-first is the
    interface both stores want. Titles with no row are simply absent, which
    the caller reads as zero.
    """
    unique = list({t for t in titles if t})
    if not unique:
        return {}

    counts: dict[str, int] = {}
    cursor = conn.cursor()
    for start in range(0, len(unique), _LOOKUP_CHUNK_SIZE):
        chunk = unique[start : start + _LOOKUP_CHUNK_SIZE]
        placeholders = ",".join("?" * len(chunk))
        cursor.execute(f"SELECT title, count FROM backlinks WHERE title IN ({placeholders})", chunk)
        for row in cursor.fetchall():
            counts[row["title"]] = row["count"]
    return counts


def rerank(
    query: str,
    candidates: list[dict],
    backlink_conn: sqlite3.Connection,
    title_weight: float,
    backlink_weight: float,
    max_backlink_count: int,
) -> list[dict]:
    """Rerank candidates based on title match and backlink count.

    max_backlink_count is a corpus-wide constant (the ~99th-percentile
    backlink count, computed once by the caller) used to normalize
    log(1 + backlink_count) into roughly the same 0-1 range as title_match,
    so backlink popularity nudges the score rather than dominating it.

    "score" is left untouched (it stays the caller's raw similarity) and the
    combined value this function ranked by is returned as "rerank_score".
    It used to be popped off before returning, which left callers holding a
    list whose order no number in it explained -- and re-sorting by "score",
    the obvious thing to do with a scored list, silently undid the
    reranking. Promoting it to the user-facing score is search.engine's job,
    not this function's.

    The sort is stable, so candidates that tie on rerank_score come back in
    the order they were passed in.
    """
    counts = _backlink_counts(backlink_conn, [c["title"] for c in candidates])
    query_set = set(query.lower().split())
    backlink_denominator = math.log(1 + max_backlink_count)

    for candidate in candidates:
        backlink_count = counts.get(candidate["title"], 0)

        title_set = set(candidate["title"].lower().split())
        title_match = len(title_set.intersection(query_set)) / len(query_set) if query_set else 0

        candidate["rerank_score"] = (
            candidate["score"]
            + title_weight * title_match
            + backlink_weight * (math.log(1 + backlink_count) / backlink_denominator)
        )

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates
