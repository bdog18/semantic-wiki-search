"""Session-wide guard: tests never see the real data root.

`swsearch.config.settings` is a module-level singleton built at import time,
and it resolves every artifact path from `paths.data_root` -- which defaults
to the repo's own `data/` directory. That directory holds ~440GB of corpus,
indexes and metadata stores that take the better part of a day to rebuild,
and several pipeline functions are destructive by design against those exact
default paths (`create_faiss_meta_db` deletes its target,
`build_backlink_counts_sqlite` drops and recreates its output table, Spark
writes with mode("overwrite")).

tests/test_pipeline.py calls the real `run_full_pipeline()` and mocks each
stage individually, so any stage added without a matching mock would run for
real against those defaults. That has happened. Redirecting data_root here
means the worst case is a stray file in a temp directory instead of a
destroyed index.

This must run before anything imports swsearch.config, which is why it sets
the environment variable at conftest import time rather than in a fixture.
"""

import os
import tempfile

_TEST_DATA_ROOT = tempfile.mkdtemp(prefix="swsearch-tests-")
os.environ["SWSEARCH_PATHS__DATA_ROOT"] = _TEST_DATA_ROOT

import pytest  # noqa: E402 -- must come after the env var is set

from swsearch.config import settings  # noqa: E402


def test_data_root_is_isolated():
    """Belt and braces: fail loudly if the redirect ever stops working."""
    assert str(settings.paths.data_root) == _TEST_DATA_ROOT


@pytest.fixture(autouse=True)
def _assert_data_root_isolated():
    repo_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    assert not str(settings.paths.data_root).startswith(repo_data), (
        f"settings.paths.data_root points at the real corpus ({settings.paths.data_root}); "
        "refusing to run tests that could write to it"
    )
    yield
