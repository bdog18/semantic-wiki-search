"""Orchestrates the full swsearch pipeline end to end.

Stages, in order:
  1. fetch raw data       (data/raw/*.sql, the XML dump, wikiextractor output --
                            skipped per-item if already present; see fetch.py)
  2. build link graph      (SQL dumps -> wiki_link_graph.db)
  3. build backlink counts (wiki_link_graph.db -> wiki_backlink_counts.db)
  4. extract               (wikiextractor output -> cleaned JSON/JSONL + article titles)
  5. embed                 (paragraphs -> embeddings + metadata + manifest)
  6. build index           (manifest -> FAISS index, alignment-checked)
  7. mine triplets (opt-in) (training data for `swsearch train-transfer`;
                              search/evaluate don't consume this)

Stage 3 is here because search.engine reads wiki_backlink_counts.db and a
missing one does not fail -- it logs a warning and silently serves unreranked
results. Reranking is the single biggest quality lever in this system (MRR
0.7947 vs 0.3767), so a pipeline that produced everything *except* the
backlink counts handed back an engine that looked like it worked and scored
half as well. `swsearch build-backlinks` still exists to rebuild it alone.

Each stage's business logic lives in its own module (linkgraph, extract, embed,
index, mining) -- this module just calls them in the right order with paths
from Settings, so it stays a thin orchestrator rather than a second copy of
any stage's logic.

Individual stages already report their own item-by-item progress via tqdm
(file counts, paragraph counts, etc.); what's missing on a multi-hour run is
knowing *which* stage is currently running, how long it's been running, and
how many stages are left. `_stage()` below logs that around each one -- a
start banner with total-elapsed-so-far, and a finish banner with that stage's
own duration -- so scrollback (or a redirected log file, where tqdm's
carriage-return bars don't render usefully anyway) tells the whole story.
"""
import time
from contextlib import contextmanager

from swsearch.config import settings
from swsearch.logutil import get_logger

logger = get_logger(__name__)

_TOTAL_STAGES = 7


def _format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


@contextmanager
def _stage(index: int, name: str, pipeline_start: float, skipped: bool = False):
    total_elapsed = _format_elapsed(time.monotonic() - pipeline_start)
    if skipped:
        logger.info("=== Stage %d/%d: %s (skipped) [pipeline elapsed: %s] ===", index, _TOTAL_STAGES, name, total_elapsed)
        yield
        return

    logger.info("=== Stage %d/%d: %s [pipeline elapsed: %s] ===", index, _TOTAL_STAGES, name, total_elapsed)
    stage_start = time.monotonic()
    try:
        yield
    finally:
        stage_elapsed = _format_elapsed(time.monotonic() - stage_start)
        total_elapsed = _format_elapsed(time.monotonic() - pipeline_start)
        remaining = _TOTAL_STAGES - index
        logger.info(
            "--- Stage %d/%d (%s) done in %s [pipeline elapsed: %s, %d stage%s remaining] ---",
            index, _TOTAL_STAGES, name, stage_elapsed, total_elapsed, remaining, "" if remaining == 1 else "s",
        )


def run_full_pipeline(
    with_triplets: bool = False,
    skip_fetch: bool = False,
    index_type: str = "auto",
) -> None:
    """index_type "auto" picks flat for small corpora and IVF-PQ for large
    ones -- see index.faiss_store.resolve_index_type. The old default was a
    hard "flat", which is right for a scratch corpus and catastrophic for the
    real one: a flat index over 42M 384-dim vectors is 64GB, built in memory
    before it is written, and it is what OOM'd earlier in this project. The
    indexes actually shipped here are IVF-PQ at 999MB.
    """
    if index_type not in ("auto", "flat", "ivfpq"):
        raise ValueError(f"index_type must be 'auto', 'flat' or 'ivfpq', got {index_type!r}")

    paths = settings.paths
    pipeline_start = time.monotonic()

    with _stage(1, "fetch raw data", pipeline_start, skipped=skip_fetch):
        if not skip_fetch:
            from swsearch.fetch import ensure_raw_data
            ensure_raw_data()

    with _stage(2, "build link graph", pipeline_start):
        from swsearch.linkgraph.build import export_link_graph_to_jsonl
        from swsearch.linkgraph.store import build_linkgraph_sqlite

        export_link_graph_to_jsonl(
            str(paths.page_sql_path),
            str(paths.pagelinks_sql_path),
            str(paths.linktarget_sql_path),
            str(paths.link_graph_jsonl_dir),
        )
        build_linkgraph_sqlite(str(paths.link_graph_jsonl_dir), str(paths.link_graph_db_path))

    with _stage(3, "build backlink counts", pipeline_start):
        from swsearch.linkgraph.backlinks import build_backlink_counts_sqlite

        build_backlink_counts_sqlite(str(paths.link_graph_db_path), str(paths.backlink_counts_db_path))

    with _stage(4, "extract/clean articles", pipeline_start):
        from swsearch.extract.wikidump import convert_json_array_to_jsonl, save_article_titles, traverse_directory

        traverse_directory(str(paths.extracted_dir), str(paths.json_dir))
        convert_json_array_to_jsonl(str(paths.json_dir), str(paths.jsonl_dir))
        save_article_titles(str(paths.jsonl_dir), str(paths.article_titles_path))

    with _stage(5, "embed paragraphs", pipeline_start):
        from swsearch.embed.paragraphs import embed_paragraphs

        embed_paragraphs(
            data_dir=str(paths.json_dir),
            embeddings_dir=str(paths.embeddings_dir),
            meta_db_path=str(paths.faiss_meta_db_path),
            model_name=settings.model.embedding_model_name,
            device=settings.model.device,
        )

    with _stage(6, f"build FAISS index ({index_type})", pipeline_start):
        from swsearch.index.faiss_store import build_index_from_manifest
        from swsearch.metadata.store import load_faiss_meta_sqlite

        meta_conn = load_faiss_meta_sqlite(str(paths.faiss_meta_db_path))
        try:
            build_index_from_manifest(
                str(paths.embeddings_dir), meta_conn, str(paths.faiss_index_path), index_type=index_type
            )
        finally:
            meta_conn.close()

    with _stage(7, "mine triplets", pipeline_start, skipped=not with_triplets):
        if with_triplets:
            from swsearch.mining.triplets import mine_triplets

            mine_triplets(
                jsonl_dir=str(paths.jsonl_dir),
                index_path=str(paths.faiss_index_path),
                meta_db_path=str(paths.faiss_meta_db_path),
                link_db_path=str(paths.link_graph_db_path),
                article_titles_path=str(paths.article_titles_path),
                out_dir=str(paths.triplets_dir),
            )

    logger.info("Pipeline complete in %s.", _format_elapsed(time.monotonic() - pipeline_start))
