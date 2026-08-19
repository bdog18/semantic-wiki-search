"""Metadata lookup backends behind one interface.

search.engine needs the same thing from either store -- given the FAISS ids a
search returned, hand back the paragraph text, its article title, and (where
the store has it) the article's curid. SQLite serves that from a local file;
DynamoDB serves it from a table, which is what makes deployment possible at
all: the metadata store is 21GB, too large to ship in a container image and
too large to want on an attached volume.

Both backends are batch-first. The engine used to call the single-row
sqlite lookup once per candidate inside its scoring loop, which is merely
wasteful against a local file (50 queries where 1 would do) but would be
~500ms of round trips against DynamoDB. get_many() is the shape both stores
actually want.
"""

from swsearch.logutil import get_logger
from swsearch.metadata.store import get_texts_and_meta_from_db

logger = get_logger(__name__)

# BatchGetItem accepts at most 100 keys per request. Paragraph rows average
# well under 1KB, so 100 items sit far inside the 16MB response ceiling and
# the key count is the only limit that binds.
_BATCH_GET_LIMIT = 100

DYNAMODB_URI_PREFIX = "dynamodb://"


class SqliteMetaStore:
    """Local-file backend. Reports no curid: the SQLite schema is
    faiss_meta(idx, text, article_title) with no URL column, which is why the
    engine still needs article_titles.json alongside it.
    """

    provides_curid = False

    def __init__(self, connection):
        self._connection = connection

    def get_many(self, indices: list[int]) -> dict[int, tuple[str, str, int | None]]:
        rows = get_texts_and_meta_from_db(self._connection, indices)
        return {idx: (text, title, None) for idx, (text, title) in rows.items()}


class DynamoMetaStore:
    """DynamoDB backend, reading the table produced by
    migrate.dynamodb_export. Items carry the curid denormalised onto every
    paragraph, so the engine can build result URLs without article_titles.json
    and its ~1GB in-memory dict.
    """

    provides_curid = True

    def __init__(self, table_name: str, region_name: str | None = None):
        import boto3  # deferred: local SQLite runs shouldn't require boto3

        self.table_name = table_name
        self._client = boto3.client("dynamodb", region_name=region_name)
        logger.info("Metadata backend: DynamoDB table %s", table_name)

    def get_many(self, indices: list[int]) -> dict[int, tuple[str, str, int | None]]:
        unique = list({int(i) for i in indices if i >= 0})
        if not unique:
            return {}

        result: dict[int, tuple[str, str, int | None]] = {}
        for start in range(0, len(unique), _BATCH_GET_LIMIT):
            chunk = unique[start : start + _BATCH_GET_LIMIT]
            keys = [{"idx": {"N": str(i)}} for i in chunk]

            # BatchGetItem is allowed to return fewer items than asked for --
            # on throttling or a 16MB overflow it reports the remainder in
            # UnprocessedKeys rather than failing. Not draining that is a
            # silent-wrong-answers bug: the query just quietly loses
            # candidates under load.
            while keys:
                response = self._client.batch_get_item(
                    RequestItems={self.table_name: {"Keys": keys}}
                )
                for item in response.get("Responses", {}).get(self.table_name, []):
                    result[int(item["idx"]["N"])] = (
                        item.get("text", {}).get("S", ""),
                        item.get("title", {}).get("S", ""),
                        int(item["curid"]["N"]) if "curid" in item else None,
                    )
                keys = response.get("UnprocessedKeys", {}).get(self.table_name, {}).get("Keys", [])
                if keys:
                    logger.debug("Retrying %d unprocessed DynamoDB keys", len(keys))
        return result


def is_dynamodb_uri(value: str | None) -> bool:
    return bool(value) and value.startswith(DYNAMODB_URI_PREFIX)


def table_name_from_uri(uri: str) -> str:
    """dynamodb://swsearch-meta -> swsearch-meta"""
    name = uri[len(DYNAMODB_URI_PREFIX):].strip("/")
    if not name:
        raise ValueError(f"Malformed DynamoDB URI (expected dynamodb://table): {uri}")
    return name
