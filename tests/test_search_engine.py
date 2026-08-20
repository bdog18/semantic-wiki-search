"""End-to-end SearchEngine behaviour against a real (tiny) FAISS index.

Everything here runs the actual retrieve -> reconstruct -> cosine -> dedupe ->
rerank path; only the encoder is stubbed, since downloading a real
SentenceTransformer would make the suite depend on the network. The index,
the metadata store and the backlinks db are all genuine.
"""

import json
import sqlite3

import numpy as np
import pytest

from swsearch.index.faiss_store import build_flat_index_from_manifest
from swsearch.metadata.store import (
    append_faiss_meta_batch,
    create_faiss_meta_db,
    record_manifest_entry,
)
from swsearch.search.engine import SearchEngine

# Query vector points straight down the first axis, so cosine against each
# indexed vector below is readable by eye.
QUERY_VECTOR = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

# Two paragraphs from "Article A" (one a near-perfect match, one weaker) and
# one from "Article B". The duplicate article is the point: search must
# collapse to the best-scoring paragraph per article.
VECTORS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],   # idx 0, Article A -- cosine 1.0
        [0.6, 0.8, 0.0, 0.0],   # idx 1, Article A -- cosine 0.6
        [0.8, 0.6, 0.0, 0.0],   # idx 2, Article B -- cosine 0.8
    ],
    dtype=np.float32,
)
TEXTS = ["A's best paragraph", "A's weaker paragraph", "B's only paragraph"]
TITLES = ["Article A", "Article A", "Article B"]


class _StubEncoder:
    """Stands in for SentenceTransformer: always returns QUERY_VECTOR."""

    def __init__(self, *args, **kwargs):
        pass

    def encode(self, sentences, **kwargs):
        return QUERY_VECTOR.copy()


@pytest.fixture
def engine_paths(tmp_path):
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    np.save(embeddings_dir / "batch_00000.npy", VECTORS)

    meta_db_path = tmp_path / "meta.db"
    meta_conn = create_faiss_meta_db(str(meta_db_path))
    append_faiss_meta_batch(meta_conn, 0, TEXTS, TITLES)
    record_manifest_entry(meta_conn, 0, "batch_00000.npy", len(TEXTS))

    index_path = tmp_path / "paragraphs.index"
    build_flat_index_from_manifest(str(embeddings_dir), meta_conn, str(index_path))
    meta_conn.close()

    titles_path = tmp_path / "article_titles.json"
    titles_path.write_text(json.dumps({
        "Article A": "https://en.wikipedia.org/wiki?curid=1",
        "Article B": "https://en.wikipedia.org/wiki?curid=2",
    }))

    return {"index": str(index_path), "meta": str(meta_db_path), "titles": str(titles_path)}


def _backlinks_db(tmp_path, counts):
    path = tmp_path / "backlinks.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE backlinks (title TEXT PRIMARY KEY, count INTEGER)")
    conn.executemany("INSERT INTO backlinks VALUES (?, ?)", counts.items())
    conn.commit()
    conn.close()
    return str(path)


def _engine(monkeypatch, paths, **kwargs):
    monkeypatch.setattr("swsearch.search.engine.SentenceTransformer", _StubEncoder)
    return SearchEngine(
        index_path=paths["index"],
        meta_db_path=paths["meta"],
        article_titles_path=paths["titles"],
        **kwargs,
    )


def test_deduplicates_to_the_best_paragraph_per_article(monkeypatch, engine_paths):
    results = _engine(monkeypatch, engine_paths, rerank_enabled=False).search("anything", k=10)

    # Three indexed paragraphs, two articles -> two results.
    assert [r["title"] for r in results] == ["Article A", "Article B"]
    # Article A's weaker paragraph must lose to its stronger one.
    assert results[0]["snippet"] == "A's best paragraph"


def test_unreranked_results_are_descending_cosine(monkeypatch, engine_paths):
    results = _engine(monkeypatch, engine_paths, rerank_enabled=False).search("anything", k=10)

    assert results[0]["score"] == pytest.approx(1.0, abs=1e-6)
    assert results[1]["score"] == pytest.approx(0.8, abs=1e-6)
    # With reranking off the two fields are the same number, but "cosine" is
    # always populated so a caller never has to branch on which mode ran.
    for result in results:
        assert result["cosine"] == pytest.approx(result["score"], abs=1e-9)


def test_k_limits_the_returned_results(monkeypatch, engine_paths):
    results = _engine(monkeypatch, engine_paths, rerank_enabled=False).search("anything", k=1)
    assert [r["title"] for r in results] == ["Article A"]


def test_urls_come_from_the_titles_map_for_the_sqlite_backend(monkeypatch, engine_paths):
    results = _engine(monkeypatch, engine_paths, rerank_enabled=False).search("anything", k=10)
    assert results[0]["url"] == "https://en.wikipedia.org/wiki?curid=1"


def test_rerank_promotes_rerank_score_and_keeps_cosine(monkeypatch, tmp_path, engine_paths):
    """The contract search.engine documents: results are ordered by "score",
    and "score" is the value they were ordered by. Getting this wrong returns
    a list sorted by a number it doesn't contain, which invites a caller to
    "fix" it by re-sorting on "score" and silently discard the rerank."""
    backlinks = _backlinks_db(tmp_path, {"Article A": 1, "Article B": 500})
    engine = _engine(monkeypatch, engine_paths, rerank_enabled=True, backlink_db_path=backlinks)

    results = engine.search("anything", k=10)

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # Cosine is carried through untouched: Article A still has the better
    # embedding match even where reranking reorders around it.
    by_title = {r["title"]: r for r in results}
    assert by_title["Article A"]["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert by_title["Article B"]["cosine"] == pytest.approx(0.8, abs=1e-6)
    # rerank_score is an internal field; it must not leak to callers.
    assert all("rerank_score" not in r for r in results)


def test_backlink_authority_can_reorder_results(monkeypatch, tmp_path, engine_paths):
    # Article B trails on cosine (0.8 vs 1.0) but is far more linked-to.
    backlinks = _backlinks_db(tmp_path, {"Article A": 0, "Article B": 10_000})
    engine = _engine(monkeypatch, engine_paths, rerank_enabled=True, backlink_db_path=backlinks)

    assert [r["title"] for r in engine.search("anything", k=10)] == ["Article B", "Article A"]


def test_per_call_flag_can_disable_reranking(monkeypatch, tmp_path, engine_paths):
    backlinks = _backlinks_db(tmp_path, {"Article A": 0, "Article B": 10_000})
    engine = _engine(monkeypatch, engine_paths, rerank_enabled=True, backlink_db_path=backlinks)

    reranked = engine.search("anything", k=10)
    raw = engine.search("anything", k=10, rerank_enabled=False)

    assert [r["title"] for r in reranked] == ["Article B", "Article A"]
    assert [r["title"] for r in raw] == ["Article A", "Article B"]


def test_missing_backlinks_db_degrades_to_unreranked(monkeypatch, tmp_path, engine_paths):
    # Documented behaviour: a missing backlinks file logs a warning and turns
    # reranking off rather than failing the whole engine.
    engine = _engine(
        monkeypatch, engine_paths,
        rerank_enabled=True,
        backlink_db_path=str(tmp_path / "does-not-exist.db"),
    )

    assert engine.backlink_conn is None
    results = engine.search("anything", k=10)
    assert [r["title"] for r in results] == ["Article A", "Article B"]
