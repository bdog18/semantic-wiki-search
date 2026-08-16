import json

import numpy as np
from sentence_transformers import SentenceTransformer

from swsearch.config import settings
from swsearch.index.faiss_store import load_index
from swsearch.logutil import get_logger
from swsearch.metadata.store import get_text_and_meta_from_db, load_faiss_meta_sqlite

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
    ):
        paths = settings.paths
        self.index = load_index(index_path or str(paths.faiss_index_path))
        self.meta_conn = load_faiss_meta_sqlite(meta_db_path or str(paths.faiss_meta_db_path))
        self.model = SentenceTransformer(model_name or settings.model.embedding_model_name)

        titles_path = article_titles_path or str(paths.article_titles_path)
        try:
            with open(titles_path, "r", encoding="utf-8") as f:
                self.article_urls: dict[str, str] = json.load(f)
        except FileNotFoundError:
            logger.warning("Article titles file not found at %s; result URLs will be empty", titles_path)
            self.article_urls = {}

    def search(self, query: str, k: int = 5, candidate_pool: int | None = None) -> list[dict]:
        """Return up to k results as [{"title", "url", "score", "snippet"}, ...],
        ranked by cosine similarity, deduplicated to one (best) hit per article.
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

        ranked = sorted(best_per_article.values(), key=lambda r: r["score"], reverse=True)
        return ranked[:k]
