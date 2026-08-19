import sqlite3
import math

from swsearch.logutil import get_logger

logger = get_logger(__name__)


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
    """
    for candidate in candidates:
        backlink_cursor = backlink_conn.cursor()
        backlink_cursor.execute("SELECT title, count FROM backlinks WHERE title = ?", (candidate["title"],))
        backlink_row = backlink_cursor.fetchone()
        backlink_count = backlink_row["count"] if backlink_row is not None else 0

        title_set = set(candidate["title"].lower().split())
        query_set = set(query.lower().split())
        title_match = len(title_set.intersection(query_set)) / len(query_set) if query_set else 0

        candidate["rerank_score"] = (
            candidate["score"]
            + title_weight * title_match
            + backlink_weight * (math.log(1 + backlink_count) / math.log(1 + max_backlink_count))
        )

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates


if __name__ == "__main__":
    from swsearch.linkgraph.backlinks import load_backlink_counts_sqlite
    from swsearch.config import settings

    # Example usage
    query = "example query"
    candidates = [
        {"title": "Example Title 1", "score": 0.8},
        {"title": "Example Title 2", "score": 0.6},
    ]
    backlink_conn = load_backlink_counts_sqlite(str(settings.paths.backlink_counts_db_path))
    title_weight = 0.5
    backlink_weight = 0.3
    max_backlink_count = 20000

    reranked_results = rerank(query, candidates, backlink_conn, title_weight, backlink_weight, max_backlink_count)
    print(reranked_results)
