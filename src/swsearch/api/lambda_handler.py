"""AWS Lambda entry point for the search API.

Mangum translates between Lambda's event/response dicts and the ASGI
interface FastAPI speaks, so app.py stays a plain FastAPI application that
still runs under uvicorn locally and on any container platform.

Two things here are not boilerplate:

  * The engine is loaded at *module scope*, not through Mangum's lifespan
    support. Lambda imports this module during the INIT phase; Mangum runs
    ASGI lifespan events on the first invocation instead. Loading here puts
    the ~10s index load in container initialisation, where it happens once
    per container, rather than inside the first user's request.

  * Warmup pings are answered before Mangum sees them. An EventBridge
    schedule event is not an HTTP event, and handing one to Mangum raises a
    KeyError on the missing request context. Intercepting it is also the
    point: a warmup should cost a few milliseconds, not a full search.
"""

import os

from mangum import Mangum

from swsearch.api.app import app, load_engine
from swsearch.logutil import get_logger

logger = get_logger(__name__)

# Marker an EventBridge scheduled rule sends to keep a container alive.
# Lambda recycles idle containers, and a cold start costs ~10s; a ping every
# few minutes costs a few cents a month and means most visitors never see
# one.
WARMUP_KEY = "swsearch_warmup"

# Runs during INIT. A failure here surfaces as an init error in CloudWatch
# rather than as a mysterious 500 on someone's first search.
load_engine()

# lifespan="off": startup already happened above, and letting Mangum run it
# again would rebuild the engine inside the first request.
_asgi_handler = Mangum(app, lifespan="off")


def handler(event, context):
    if isinstance(event, dict) and event.get(WARMUP_KEY):
        logger.debug("Warmup ping")
        return {"statusCode": 200, "body": '{"status":"warm"}'}
    return _asgi_handler(event, context)
