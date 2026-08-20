import importlib

import pytest

# Every importable module in the package. This is the cheapest guard there
# is against a module that stops importing -- a bad relative import, a
# dependency that moved, a package directory that lost its __init__.py and
# stopped being shipped in a built wheel. Add new modules here.
#
# swsearch.api.lambda_handler is deliberately absent: it imports mangum
# (Lambda-image only) and loads the index at import time. tests/
# test_lambda_handler.py covers it with both stubbed out.
MODULES = [
    "swsearch",
    "swsearch.config",
    "swsearch.logutil",
    "swsearch.cli",
    "swsearch.common.spark",
    "swsearch.common.textsplit",
    "swsearch.extract.wikidump",
    "swsearch.extract.wikitext",
    "swsearch.linkgraph.build",
    "swsearch.linkgraph.store",
    "swsearch.linkgraph.backlinks",
    "swsearch.embed.paragraphs",
    "swsearch.metadata.store",
    "swsearch.metadata.backends",
    "swsearch.index.faiss_store",
    "swsearch.mining.triplets",
    "swsearch.rerank.heuristic",
    "swsearch.search.engine",
    "swsearch.eval.metrics",
    "swsearch.tools.inspect_index",
    "swsearch.migrate.dynamodb_export",
    "swsearch.api.app",
    "swsearch.api.artifacts",
    "swsearch.fetch",
    "swsearch.pipeline",
    "swsearch.train",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
