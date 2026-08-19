import json

import numpy as np
from sentence_transformers import SentenceTransformer

from swsearch.config import settings
from swsearch.index.faiss_store import load_index
from swsearch.logutil import get_logger
from swsearch.metadata.backends import (
    DynamoMetaStore,
    SqliteMetaStore,
    is_dynamodb_uri,
    table_name_from_uri,
)
from swsearch.metadata.store import load_faiss_meta_sqlite
from swsearch.rerank.heuristic import rerank
from swsearch.linkgraph.backlinks import load_backlink_counts_sqlite

logger = get_logger(__name__)


class SearchEngine:
    """Embed a query, retrieve FAISS candidates, rerank by cosine similarity
    against the reconstructed candidate vectors, and collapse to one
    best-scoring result per article.

    This is baseline.ipynb's proven embed -> flat-FAISS -> cosine-rerank
    recipe, generalized from whole-article embeddings to the paragraph-level
    index/metadata the rest of the pipeline shares with triplet mining: many
    paragraphs can map to the same article, so results are deduplicated by
    article title before the top-k cut.
    """

    def __init__(
        self,
        index_path: str | None = None,
        meta_db_path: str | None = None,
        article_titles_path: str | None = None,
        model_name: str | None = None,
        rerank_enabled: bool | None = None,
        backlink_db_path: str | None = None,
        index_mmap: bool = False,
    ):
        paths = settings.paths
        self.index = load_index(index_path or str(paths.faiss_index_path), mmap=index_mmap)

        # meta_db_path doubles as a backend selector: "dynamodb://table"
        # picks the remote store, anything else is a local SQLite file.
        meta_target = meta_db_path or str(paths.faiss_meta_db_path)
        if is_dynamodb_uri(meta_target):
            self.meta_conn = None
            self.meta_store = DynamoMetaStore(table_name_from_uri(meta_target))
        else:
            self.meta_conn = load_faiss_meta_sqlite(meta_target)
            self.meta_store = SqliteMetaStore(self.meta_conn)

        self.model = SentenceTransformer(model_name or settings.model.embedding_model_name)
        self.rerank_enabled = rerank_enabled if rerank_enabled is not None else settings.rerank.enabled

        # article_titles.json is a ~300MB file that inflates to roughly 1GB as
        # a dict, and exists only to turn a title into a URL. A backend that
        # carries the curid on each row answers that from the row itself, so
        # the file is loaded only for backends that don't.
        self.article_urls: dict[str, str] = {}
        if not self.meta_store.provides_curid:
            titles_path = article_titles_path or str(paths.article_titles_path)
            try:
                with open(titles_path, "r", encoding="utf-8") as f:
                    self.article_urls = json.load(f)
            except FileNotFoundError:
                logger.warning("Article titles file not found at %s; result URLs will be empty", titles_path)
        
        self.backlink_conn = None
        self.max_backlink_count = 1
        # Explicit override for the same reason index_path and meta_db_path
        # take one: PathSettings derives this from data_root, so a container
        # that bakes the database in at a fixed location has no other way to
        # say where it put it.
        backlinks_path = backlink_db_path or str(settings.paths.backlink_counts_db_path)
        if self.rerank_enabled:
            try:
                self.backlink_conn = load_backlink_counts_sqlite(backlinks_path)
                # A property of the corpus, so it is configuration rather than
                # something to rediscover on every boot: the query below sorts
                # the whole 1.3GB backlinks table and measured 23.2s, which was
                # the bulk of this service's cold start. settings pins the
                # answer; set max_backlink_count to 0 to recompute it (after
                # rebuilding the backlink counts, say).
                self.max_backlink_count = settings.rerank.max_backlink_count
                if not self.max_backlink_count:
                    logger.info("Computing max_backlink_count (slow: full table sort)")
                    cur = self.backlink_conn.cursor()
                    cur.execute(
                        "SELECT count FROM backlinks ORDER BY count DESC "
                        "LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.01 AS INT) FROM backlinks)"
                    )
                    row = cur.fetchone()
                    self.max_backlink_count = row["count"] if row is not None else 1
            except FileNotFoundError:
                logger.warning("Backlink counts database not found at %s; backlink reranking will be disabled", backlinks_path)
                self.backlink_conn = None

    def _url_for(self, title: str, curid: int | None) -> str:
        """Page-id URLs where the backend supplies a curid, falling back to
        the title->URL map otherwise. Page ids are used rather than
        /wiki/{Title} because they survive article renames.
        """
        if curid is not None:
            return f"https://en.wikipedia.org/wiki?curid={curid}"
        return self.article_urls.get(title, "")

    def search(
        self,
        query: str,
        k: int = 5,
        candidate_pool: int | None = None,
        rerank_enabled: bool | None = None,
    ) -> list[dict]:
        """Return up to k results as [{"title", "url", "score", "snippet"}, ...],
        ranked by cosine similarity, deduplicated to one (best) hit per article.

        rerank_enabled overrides this instance's default for a single call
        (e.g. to let a caller compare reranked vs. raw results); it can only
        disable reranking, not enable it if the backlink DB wasn't loaded.
        """
        pool = candidate_pool or max(k * 10, 50)
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype(np.float32), pool)

        query_vec = query_embedding[0]
        query_norm = np.linalg.norm(query_vec) or 1.0

        # One metadata round trip for the whole candidate pool, not one per
        # candidate. Against SQLite that is 1 query instead of ~50; against
        # DynamoDB it is the difference between one BatchGetItem and 50
        # sequential network calls, which is most of a second per search.
        candidates = [(rank_pos, int(idx)) for rank_pos, idx in enumerate(indices[0]) if idx >= 0]
        metadata = self.meta_store.get_many([idx for _, idx in candidates])

        best_per_article: dict[str, dict] = {}
        for rank_pos, idx in candidates:
            row = metadata.get(idx)
            if row is None:
                continue
            text, title, curid = row
            if not text or not title:
                continue

            try:
                candidate_vec = self.index.reconstruct(idx)
                candidate_norm = np.linalg.norm(candidate_vec) or 1.0
                score = float(np.dot(query_vec, candidate_vec) / (query_norm * candidate_norm))
            except RuntimeError:
                # Index doesn't support reconstruction (e.g. IVF-PQ without a
                # direct map); fall back to FAISS's own distance, negated so
                # that higher is still better.
                score = -float(distances[0][rank_pos])

            existing = best_per_article.get(title)
            if existing is None or score > existing["score"]:
                best_per_article[title] = {
                    "title": title,
                    "url": self._url_for(title, curid),
                    "score": score,
                    "snippet": text,
                }

        use_rerank = self.backlink_conn is not None and rerank_enabled is not False
        if use_rerank:
            ranked = rerank(
                query,
                list(best_per_article.values()),
                self.backlink_conn,
                title_weight=settings.rerank.title_match_weight,
                backlink_weight=settings.rerank.backlink_weight,
                max_backlink_count=self.max_backlink_count,
            )
            # Promote the value the results were actually ordered by into
            # "score", keeping the cosine as "cosine". Without this the
            # returned list is sorted by a number it doesn't contain: a
            # reranked response comes back 0.738, 0.710, 0.732, which reads
            # as a bug and invites a caller to "fix" it by re-sorting on
            # "score" -- discarding the rerank entirely. Post-condition
            # either way: results are in descending "score" order.
            for result in ranked:
                result["cosine"] = result["score"]
                result["score"] = result.pop("rerank_score")
        else:
            ranked = sorted(best_per_article.values(), key=lambda r: r["score"], reverse=True)
            for result in ranked:
                result["cosine"] = result["score"]

        return ranked[:k]
