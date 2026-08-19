import sqlite3

from swsearch.rerank.heuristic import rerank


def _backlink_conn(tmp_path, counts: dict[str, int]):
    """In-memory-style backlinks db, same shape as
    linkgraph.backlinks.load_backlink_counts_sqlite produces."""
    db_path = str(tmp_path / "backlinks.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE backlinks (title TEXT PRIMARY KEY, count INTEGER)")
    conn.executemany("INSERT INTO backlinks (title, count) VALUES (?, ?)", counts.items())
    conn.commit()
    conn.row_factory = sqlite3.Row
    return conn


def _candidate(title, score, url="", snippet=""):
    return {"title": title, "url": url, "score": score, "snippet": snippet}


# Stands in for the corpus-wide ~99th-percentile backlink count that
# search.engine.SearchEngine computes once per instance. It's the denominator
# of rerank's log normalization, so it has to be passed explicitly and has to
# be non-zero -- 100 keeps the arithmetic easy to reason about here: an
# article with exactly 100 backlinks normalizes to a boost of 1.0.
_MAX_BACKLINKS = 100


def test_rerank_boosts_title_match(tmp_path):
    conn = _backlink_conn(tmp_path, {})
    candidates = [
        _candidate("Unrelated Article", score=0.9),
        _candidate("Python Programming", score=0.5),
    ]

    ranked = rerank("python programming", candidates, conn, title_weight=1.0, backlink_weight=0.0, max_backlink_count=_MAX_BACKLINKS)

    assert [c["title"] for c in ranked] == ["Python Programming", "Unrelated Article"]


def test_rerank_boosts_backlink_count(tmp_path):
    conn = _backlink_conn(tmp_path, {"Popular Article": 100, "Obscure Article": 0})
    candidates = [
        _candidate("Obscure Article", score=0.5),
        _candidate("Popular Article", score=0.49),
    ]

    ranked = rerank("query", candidates, conn, title_weight=0.0, backlink_weight=1.0, max_backlink_count=_MAX_BACKLINKS)

    assert [c["title"] for c in ranked] == ["Popular Article", "Obscure Article"]


def test_rerank_missing_title_defaults_to_zero_backlinks(tmp_path):
    # "Missing Title" has no row in backlinks at all -- must not raise, and
    # must resolve to the same backlink_count=0 as an explicit 0 row.
    conn = _backlink_conn(tmp_path, {"Explicit Zero": 0})
    candidates = [
        _candidate("Missing Title", score=0.5),
        _candidate("Explicit Zero", score=0.5),
    ]

    ranked = rerank("query", candidates, conn, title_weight=0.0, backlink_weight=1.0, max_backlink_count=_MAX_BACKLINKS)

    # Equal base scores and both should get zero backlink boost (log(1+0)=0),
    # so they tie on combined score -- list.sort() is stable, so a tie must
    # preserve input order. If the missing-row path resolved to anything
    # other than 0, this ordering would flip.
    assert [c["title"] for c in ranked] == ["Missing Title", "Explicit Zero"]


def test_rerank_preserves_original_score_field(tmp_path):
    conn = _backlink_conn(tmp_path, {})
    candidates = [_candidate("A", score=0.42)]

    ranked = rerank("query", candidates, conn, title_weight=0.5, backlink_weight=0.5, max_backlink_count=_MAX_BACKLINKS)

    assert ranked[0]["score"] == 0.42


def test_rerank_adds_rerank_score_and_preserves_other_fields(tmp_path):
    # rerank_score is now returned rather than popped -- search.engine
    # promotes it to the user-facing "score" so the order results come back
    # in is the order their score implies. With both weights at zero it
    # reduces to the input score exactly.
    conn = _backlink_conn(tmp_path, {})
    candidates = [_candidate("A", score=0.1, url="http://x", snippet="hello")]

    ranked = rerank("query", candidates, conn, title_weight=0.0, backlink_weight=0.0, max_backlink_count=_MAX_BACKLINKS)

    assert ranked == [
        {"title": "A", "url": "http://x", "score": 0.1, "snippet": "hello", "rerank_score": 0.1}
    ]


def test_rerank_score_ranks_above_raw_score(tmp_path):
    # The ordering guarantee search.engine depends on: whatever rerank
    # reorders, rerank_score is descending across the returned list even
    # when the raw scores are not.
    conn = _backlink_conn(tmp_path, {"Popular Article": 100})
    candidates = [
        _candidate("Obscure Article", score=0.9),
        _candidate("Popular Article", score=0.5),
    ]

    ranked = rerank("query", candidates, conn, title_weight=0.0, backlink_weight=1.0, max_backlink_count=_MAX_BACKLINKS)

    scores = [c["rerank_score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert [c["score"] for c in ranked] != sorted([c["score"] for c in ranked], reverse=True)


def test_rerank_empty_candidates(tmp_path):
    conn = _backlink_conn(tmp_path, {})

    assert rerank("query", [], conn, title_weight=1.0, backlink_weight=1.0, max_backlink_count=_MAX_BACKLINKS) == []
