# Lambda build of the search API. Same FastAPI app as the App Runner image,
# fronted by Mangum, with every artifact baked in.
#
#   docker build -f docker/lambda.Dockerfile -t swsearch-lambda .
#
# Everything is in the image because Lambda has no durable local disk: /tmp
# is 512MB by default and is discarded with the container, so an S3 download
# would repeat on every cold start. That trades image size (~6GB, under the
# 10GB limit) for a cold start that stays near 10s.
FROM public.ecr.aws/lambda/python:3.12

# torch from PyTorch's CPU index: the PyPI default bundles ~800MB of CUDA
# libraries that Lambda has no device for. Its own layer, so the slowest
# download in the build is also the most cacheable.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

COPY docker/lambda-requirements.txt .
RUN pip install --no-cache-dir -r lambda-requirements.txt

# Saved to an explicit directory and loaded by path, rather than left in the
# HuggingFace cache and looked up by name. Lambda's filesystem is read-only
# outside /tmp, and a cache lookup wants to take a file lock in the cache
# directory; loading a plain local path never consults the hub at all.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2').save('/opt/swsearch/encoder')"
ENV SWSEARCH_API_MODEL_NAME=/opt/swsearch/encoder

# Large, stable layers before the code, so editing src/ doesn't re-copy 2.3GB.
COPY data/processed/wiki_backlink_counts.db /opt/swsearch/wiki_backlink_counts.db
COPY data/processed/models/baseline/runs/all-MiniLM-L6/paragraphs.index /opt/swsearch/paragraphs.index
ENV SWSEARCH_API_BACKLINK_DB_PATH=/opt/swsearch/wiki_backlink_counts.db
ENV SWSEARCH_API_INDEX_PATH=/opt/swsearch/paragraphs.index

# Metadata is the one artifact that stays remote -- 21GB doesn't fit in an
# image. Overridable so a local RIE run can point at the SQLite file instead.
ENV SWSEARCH_API_META_DB_PATH=dynamodb://swsearch-meta

COPY pyproject.toml .
COPY src/ ./src/
# --no-deps: lambda-requirements.txt is the authority. A plain install would
# resolve pyproject's dependencies and pull pyspark, datasets and gradio in.
RUN pip install --no-cache-dir --no-deps .

CMD ["swsearch.api.lambda_handler.handler"]
