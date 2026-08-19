import importlib
import sys
from unittest import mock

import pytest

# mangum only ships in the Lambda image; skip rather than force it into the
# development environment, which never runs this path.
pytest.importorskip("mangum")


@pytest.fixture
def handler_module():
    """Import lambda_handler with the engine load stubbed out.

    The module calls load_engine() at import time on purpose -- that is what
    puts the index load in Lambda's INIT phase -- so importing it for real
    here would read a 1GB index.
    """
    sys.modules.pop("swsearch.api.lambda_handler", None)
    with mock.patch("swsearch.api.app.load_engine") as load:
        module = importlib.import_module("swsearch.api.lambda_handler")
        module._loaded_engine_mock = load
        yield module
    sys.modules.pop("swsearch.api.lambda_handler", None)


def test_engine_is_loaded_at_import(handler_module):
    # If this regresses to lazy loading, the index load moves out of INIT and
    # into the first user's request.
    handler_module._loaded_engine_mock.assert_called_once()


def test_warmup_event_short_circuits(handler_module):
    with mock.patch.object(handler_module, "_asgi_handler") as asgi:
        response = handler_module.handler({handler_module.WARMUP_KEY: True}, None)

    assert response["statusCode"] == 200
    # The point of a warmup is that it costs nothing: an EventBridge ping
    # must not be translated into an HTTP request and run a real search.
    asgi.assert_not_called()


def test_http_event_is_delegated_to_mangum(handler_module):
    event = {"version": "2.0", "rawPath": "/health", "requestContext": {"http": {"method": "GET"}}}
    with mock.patch.object(handler_module, "_asgi_handler", return_value={"statusCode": 200}) as asgi:
        handler_module.handler(event, None)

    asgi.assert_called_once_with(event, None)


def test_non_dict_event_is_delegated(handler_module):
    # Defensive: .get() on a non-dict would raise before Mangum ever saw it.
    with mock.patch.object(handler_module, "_asgi_handler", return_value={}) as asgi:
        handler_module.handler([], None)
    asgi.assert_called_once()
