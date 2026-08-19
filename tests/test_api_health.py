from unittest import mock

import pytest
from fastapi.testclient import TestClient

import swsearch.api.app as api


class _Engine:
    """Minimal stand-in: health only touches the index count and the store."""

    def __init__(self, store):
        self.index = mock.Mock(ntotal=41953396)
        self.meta_store = store


@pytest.fixture
def client():
    # TestClient only runs lifespan as a context manager, so _engine stays
    # whatever we set rather than loading a real 1GB index.
    return TestClient(api.app)


def test_health_503_when_engine_absent(client):
    with mock.patch.object(api, "_engine", None):
        assert client.get("/health").status_code == 503


def test_health_ok_reports_backend_and_probe(client):
    store = mock.Mock(get_many=mock.Mock(return_value={0: ("text", "Title", 42)}))
    with mock.patch.object(api, "_engine", _Engine(store)):
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["vectors"] == 41953396
    assert body["metadata_probe_hit"] is True
    store.get_many.assert_called_once_with([0])


def test_health_ok_when_probe_returns_nothing(client):
    # A successful query that finds no row is still a working backend.
    store = mock.Mock(get_many=mock.Mock(return_value={}))
    with mock.patch.object(api, "_engine", _Engine(store)):
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["metadata_probe_hit"] is False


def test_health_503_when_metadata_backend_raises(client):
    # The case that motivated this: a table that doesn't exist yet. Without
    # the probe this returned 200 and failed on the first real search.
    store = mock.Mock(get_many=mock.Mock(side_effect=RuntimeError("ResourceNotFoundException")))
    with mock.patch.object(api, "_engine", _Engine(store)):
        response = client.get("/health")

    assert response.status_code == 503
    assert "Metadata backend unavailable" in response.json()["detail"]
