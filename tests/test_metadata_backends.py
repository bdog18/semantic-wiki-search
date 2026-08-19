import sqlite3

import pytest

from swsearch.metadata.backends import (
    DynamoMetaStore,
    SqliteMetaStore,
    is_dynamodb_uri,
    table_name_from_uri,
)


def _sqlite_conn(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "meta.db"))
    conn.execute("CREATE TABLE faiss_meta (idx INTEGER PRIMARY KEY, text TEXT NOT NULL, article_title TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO faiss_meta (idx, text, article_title) VALUES (?, ?, ?)",
        [(1, "first para", "Alpha"), (2, "second para", "Beta")],
    )
    conn.commit()
    conn.row_factory = sqlite3.Row
    return conn


def test_uri_detection():
    assert is_dynamodb_uri("dynamodb://swsearch-meta")
    assert not is_dynamodb_uri("/data/paragraphs.index.meta.db")
    assert not is_dynamodb_uri(None)
    assert table_name_from_uri("dynamodb://swsearch-meta") == "swsearch-meta"
    assert table_name_from_uri("dynamodb://swsearch-meta/") == "swsearch-meta"
    with pytest.raises(ValueError):
        table_name_from_uri("dynamodb://")


def test_sqlite_backend_returns_none_curid(tmp_path):
    store = SqliteMetaStore(_sqlite_conn(tmp_path))
    assert store.get_many([1, 2]) == {
        1: ("first para", "Alpha", None),
        2: ("second para", "Beta", None),
    }
    assert store.provides_curid is False


def test_sqlite_backend_skips_missing_and_negative(tmp_path):
    store = SqliteMetaStore(_sqlite_conn(tmp_path))
    assert set(store.get_many([1, -1, 999])) == {1}
    assert store.get_many([]) == {}


class _FakeDynamoClient:
    """Records requests and replays scripted responses, so the batching and
    UnprocessedKeys logic can be tested without boto3 or a real table."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def batch_get_item(self, RequestItems):
        table, spec = next(iter(RequestItems.items()))
        self.requests.append([int(k["idx"]["N"]) for k in spec["Keys"]])
        return self._responses.pop(0)


def _store_with(responses, table="t"):
    store = DynamoMetaStore.__new__(DynamoMetaStore)   # bypass boto3 in __init__
    store.table_name = table
    store._client = _FakeDynamoClient(responses)
    return store


def _item(idx, text, title, curid=None):
    item = {"idx": {"N": str(idx)}, "text": {"S": text}, "title": {"S": title}}
    if curid is not None:
        item["curid"] = {"N": str(curid)}
    return item


def test_dynamo_backend_parses_items():
    store = _store_with([{"Responses": {"t": [_item(1, "para", "Alpha", 42)]}}])
    assert store.get_many([1]) == {1: ("para", "Alpha", 42)}
    assert store.provides_curid is True


def test_dynamo_backend_handles_item_without_curid():
    store = _store_with([{"Responses": {"t": [_item(1, "para", "Alpha")]}}])
    assert store.get_many([1]) == {1: ("para", "Alpha", None)}


def test_dynamo_backend_chunks_at_100_keys():
    # 250 keys must become 3 requests: BatchGetItem rejects more than 100.
    responses = [{"Responses": {"t": []}} for _ in range(3)]
    store = _store_with(responses)
    store.get_many(list(range(250)))
    assert [len(r) for r in store._client.requests] == [100, 100, 50]


def test_dynamo_backend_drains_unprocessed_keys():
    # First call returns one item and defers the other; not retrying would
    # silently drop a candidate rather than fail.
    responses = [
        {"Responses": {"t": [_item(1, "one", "Alpha")]},
         "UnprocessedKeys": {"t": {"Keys": [{"idx": {"N": "2"}}]}}},
        {"Responses": {"t": [_item(2, "two", "Beta")]}},
    ]
    store = _store_with(responses)
    assert store.get_many([1, 2]) == {
        1: ("one", "Alpha", None),
        2: ("two", "Beta", None),
    }
    assert store._client.requests == [[1, 2], [2]]


def test_dynamo_backend_skips_negative_and_empty():
    store = _store_with([])
    assert store.get_many([]) == {}
    assert store.get_many([-1]) == {}
