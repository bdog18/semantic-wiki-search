"""Stage serving artifacts from S3 onto local disk at container start.

config.PathSettings can't express "wherever the container put it": its
faiss_index_path and faiss_meta_db_path are computed properties hanging off
data_root, not settable fields, so there is no environment variable that
points the engine at a downloaded file. This module fills that gap -- it
resolves an s3:// URI to a local path, downloading once, and passes anything
else through untouched so local runs behave exactly as before.

Downloading at boot rather than baking the index into the image keeps the
image small and lets a new index roll out by restarting the service instead
of rebuilding it. The cost is a slower cold start, which is why the API's
/health reports 503 until this finishes.
"""

import os
from pathlib import Path

from swsearch.logutil import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(os.environ.get("SWSEARCH_ARTIFACT_CACHE", "/tmp/swsearch-artifacts"))


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri[len("s3://"):]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key:
        raise ValueError(f"Malformed S3 URI (expected s3://bucket/key): {uri}")
    return bucket, key


def stage(uri: str | None) -> str | None:
    """Return a local filesystem path for `uri`, downloading it if it's on S3.

    None and non-S3 paths pass straight through, so the same call site works
    unchanged in development. Re-uses an already-downloaded file: App Runner
    restarts a container far more often than the index changes, and a ~1GB
    re-download on every restart is minutes of cold start for nothing.
    """
    if not uri or not uri.startswith("s3://"):
        return uri

    import boto3  # deferred: local runs shouldn't need boto3 installed

    bucket, key = _parse_s3_uri(uri)
    destination = CACHE_DIR / bucket / key
    if destination.exists():
        logger.info("Using cached artifact %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
        return str(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temporary name and rename into place: a container killed
    # mid-download would otherwise leave a truncated file that every
    # subsequent start treats as a valid cache hit.
    partial = destination.with_suffix(destination.suffix + ".partial")
    logger.info("Downloading %s -> %s", uri, destination)
    boto3.client("s3").download_file(bucket, key, str(partial))
    partial.rename(destination)
    logger.info("Downloaded %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
    return str(destination)
