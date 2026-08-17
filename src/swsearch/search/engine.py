import json

import numpy as np
from sentence_transformers import SentenceTransformer

from swsearch.config import settings
from swsearch.index.faiss_store import load_index
from swsearch.logutil import get_logger
from swsearch.metadata.store import get_text_and_meta_from_db, load_faiss_meta_sqlite
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
    ):
        paths = settings.paths
        self.index = load_index(index_path or str(paths.faiss_index_path))
        self.meta_conn = load_faiss_meta_sqlite(meta_db_path or str(paths.faiss_meta_db_path))
        self.model = SentenceTransformer(model_name or settings.model.embedding_model_name)
        self.rerank_enabled = rerank_enabled if rerank_enabled is not None else settings.rerank.enabled
        titles_path = article_titles_path or str(paths.article_titles_path)
        
        try:
            with open(titles_path, "r", encoding="utf-8") as f:
                self.article_urls: dict[str, str] = json.load(f)
        except FileNotFoundError:
            logger.warning("Article titles file not found at %s; result URLs will be empty", titles_path)
            self.article_urls = {}
        
        self.backlink_conn = None
        self.max_backlink_count = 1
        if self.rerank_enabled:
            try:
                self.backlink_conn = load_backlink_counts_sqlite(str(settings.paths.backlink_counts_db_path))
                # Corpus-wide constant -- computed once here, not per query/candidate,
                # since the answer never changes for the lifetime of this SearchEngine.
                cur = self.backlink_conn.cursor()
                cur.execute(
                    "SELECT count FROM backlinks ORDER BY count DESC "
                    "LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.01 AS INT) FROM backlinks)"
                )
                row = cur.fetchone()
                self.max_backlink_count = row["count"] if row is not None else 1
            except FileNotFoundError:
                logger.warning("Backlink counts database not found at %s; backlink reranking will be disabled", str(settings.paths.backlink_counts_db_path))
                self.backlink_conn = None

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

        best_per_article: dict[str, dict] = {}
        for rank_pos, idx in enumerate(indices[0]):
            if idx < 0:
                continue
            text, title = get_text_and_meta_from_db(self.meta_conn, int(idx))
            if not text or not title:
                continue

            try:
                candidate_vec = self.index.reconstruct(int(idx))
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
                    "url": self.article_urls.get(title, ""),
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
        else:
            ranked = sorted(best_per_article.values(), key=lambda r: r["score"], reverse=True)
            
        return ranked[:k]
