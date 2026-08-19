# Build from the repo root so src/ is in context:
#   docker build -f docker/api.Dockerfile -t swsearch-api .
FROM python:3.12-slim

WORKDIR /app

# torch first, and from PyTorch's CPU index. The default PyPI wheel bundles
# the nvidia-*-cu12 CUDA libraries -- around 800MB of GPU runtime that a
# CPU-only FAISS service will never call. Installed as its own layer so the
# slowest download in the build is also the most cacheable.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

COPY docker/api-requirements.txt .
RUN pip install --no-cache-dir -r api-requirements.txt

# Bake the encoder into the image rather than letting SentenceTransformer
# fetch it from the hub on first use: otherwise every cold start depends on
# huggingface.co being reachable, and a container that cannot start because
# someone else's CDN is down is a needless outage.
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Backlink counts for reranking, ~1.3GB. Baked in rather than staged from S3
# at boot: rerank is what produces the 0.7947 MRR the app advertises, the
# lookup runs once per candidate per query (up to 50 per search), and a
# missing file degrades silently to unreranked results rather than erroring.
# Copied before src/ so editing code doesn't re-copy a gigabyte.
COPY data/processed/wiki_backlink_counts.db /app/data/wiki_backlink_counts.db
ENV SWSEARCH_API_BACKLINK_DB_PATH=/app/data/wiki_backlink_counts.db

COPY pyproject.toml .
COPY src/ ./src/
# --no-deps: api-requirements.txt above is the authority on what gets
# installed. A plain install would resolve pyproject's `dependencies` and
# drag pyspark, datasets, accelerate and gradio into a serving image.
RUN pip install --no-cache-dir --no-deps .

# App Runner sets PORT; this is the local-run default.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn swsearch.api.app:app --host 0.0.0.0 --port ${PORT}"]
