import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from swsearch.api.artifacts import stage
from swsearch.logutil import get_logger
from swsearch.search.engine import SearchEngine

logger = get_logger(__name__)

# Unset => no auth check, so local development needs no setup. Set it in the
# deployment's environment and the header becomes mandatory; the same code
# runs both places. Consumed here rather than in config.Settings because
# Settings is the pipeline's configuration and this is a deployment secret.
API_KEY = os.environ.get("SWSEARCH_API_KEY")

# Serving reads its artifacts from wherever they were staged (a local path in
# development, whatever the container downloaded them to in deployment),
# which config.PathSettings can't express -- faiss_index_path and
# faiss_meta_db_path are computed properties hanging off data_root, not
# settable fields. Falling through to None hands SearchEngine its own
# settings-derived defaults, so an unconfigured run behaves exactly as the
# CLI does. Settings ignores unrecognised SWSEARCH_* vars (extra="ignore"),
# so these names don't collide with it.
INDEX_PATH = os.environ.get("SWSEARCH_API_INDEX_PATH")
META_DB_PATH = os.environ.get("SWSEARCH_API_META_DB_PATH")
MODEL_NAME = os.environ.get("SWSEARCH_API_MODEL_NAME")
# Baked into the image rather than fetched at boot: it is 1.3GB, reranking is
# not optional for result quality (reranked MRR 0.7947), and the engine
# silently degrades to unreranked results if it is missing -- a failure mode
# that looks like "search got worse" rather than like an error.
BACKLINK_DB_PATH = os.environ.get("SWSEARCH_API_BACKLINK_DB_PATH")
# Trades heap for reclaimable page cache when loading the index. Off by
# default; set where the sandbox has a hard memory ceiling it will kill the
# process for crossing. An environment variable rather than a build-time
# constant so it can be flipped without rebuilding a 6.8GB image.
INDEX_MMAP = os.environ.get("SWSEARCH_API_INDEX_MMAP", "").lower() in ("1", "true", "yes")

_engine: SearchEngine | None = None


def load_engine() -> SearchEngine:
    """Build the SearchEngine, or return the one already built.

    Split out of lifespan so Lambda can call it during the INIT phase. Mangum
    runs ASGI lifespan events on the first *invocation*, which would put a
    ~10s index load inside a user's request; importing this module and
    calling load_engine() at module scope moves that work into container
    initialisation instead, where it happens once per container rather than
    once per cold request.

    Idempotent, so the uvicorn path (lifespan) and the Lambda path (module
    import) can both call it without loading the index twice.
    """
    global _engine
    if _engine is not None:
        return _engine

    # Defaults resolve to the baseline (off-the-shelf all-MiniLM-L6-v2) run,
    # the same index `swsearch search` uses. On the clean corpus it still
    # edges out the fine-tuned lr5e-6_steps8000 run on the metrics that
    # decide what a user sees first -- reranked MRR 0.7947 vs 0.7444 and
    # Top-1 0.7032 vs 0.6065 -- even though the fine-tune is slightly ahead
    # on Top-5/Top-10 and MAP. Point the SWSEARCH_API_* vars at the transfer
    # run's index/meta/model to compare.
    logger.info("Loading SearchEngine...")
    # stage() passes local paths through untouched and downloads s3:// URIs,
    # so the same three variables configure a laptop and a container.
    _engine = SearchEngine(
        index_path=stage(INDEX_PATH),
        meta_db_path=stage(META_DB_PATH),
        model_name=MODEL_NAME,
        rerank_enabled=True,
        backlink_db_path=stage(BACKLINK_DB_PATH),
        index_mmap=INDEX_MMAP,
    )
    logger.info("SearchEngine ready: %d vectors", _engine.index.ntotal)
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index once, at startup, instead of per request.

    Reading a ~1GB IVF-PQ index and building its direct map takes seconds,
    so there is a real window where the process is up but cannot answer
    anything -- which is what /health exists to report. Left as a hard
    failure on purpose: a container that starts "successfully" without a
    backend and serves 503s to every request is harder to diagnose than one
    that refuses to start.
    """
    global _engine
    load_engine()
    yield
    _engine = None


app = FastAPI(title="swsearch API", lifespan=lifespan)


class SearchRequest(BaseModel):
    # The bounds are cost control, not politeness: k and query length are the
    # two knobs a caller could otherwise use to turn one HTTP request into an
    # arbitrarily expensive metadata lookup. k's ceiling matches the Gradio
    # app's slider maximum.
    query: str = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=10)
    rerank: bool = True


class SearchResult(BaseModel):
    # Results arrive in descending "score" order. When reranking is on that
    # is the combined title/backlink score, not the raw similarity, so
    # "cosine" carries the underlying embedding distance separately -- keep
    # them distinct or a client that sorts on the wrong one throws the
    # reranking away.
    title: str
    url: str
    score: float
    cosine: float
    snippet: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


@app.get("/health")
def health() -> dict:
    """Readiness, including a probe of the metadata backend.

    Checking only that the index loaded is not readiness: the index and the
    paragraph metadata are separate systems, and DynamoMetaStore's
    constructor builds a boto3 client without contacting AWS. A deployment
    pointed at a table that doesn't exist yet -- or that it lacks permission
    to read -- would otherwise report a healthy 200 and fail on the first
    real search instead.

    One point read per call, which is negligible against the cost of finding
    out the hard way. An empty result is still healthy: it means the query
    succeeded and that row simply isn't present.
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    try:
        probe = _engine.meta_store.get_many([0])
    except Exception as exc:  # noqa: BLE001 -- a health check reports, it doesn't discriminate
        logger.warning("Metadata backend probe failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Metadata backend unavailable: {type(exc).__name__}",
        ) from exc

    return {
        "status": "ok",
        "vectors": _engine.index.ntotal,
        "metadata_backend": type(_engine.meta_store).__name__,
        "metadata_probe_hit": bool(probe),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, x_api_key: str | None = Header(default=None)) -> SearchResponse:
    """Sync on purpose: SearchEngine.search() is blocking CPU work (encode ->
    FAISS -> metadata -> rerank). Declared `async def`, it would run on the
    event loop and serialise every concurrent request behind the current
    one; as a plain `def`, FastAPI hands it to the threadpool instead.
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    results = _engine.search(req.query, k=req.k, rerank_enabled=req.rerank)
    return SearchResponse(results=results)
