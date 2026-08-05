import importlib

import pytest

MODULES = [
    "swsearch",
    "swsearch.config",
    "swsearch.logutil",
    "swsearch.cli",
    "swsearch.common.spark",
    "swsearch.common.textsplit",
    "swsearch.extract.wikidump",
    "swsearch.linkgraph.build",
    "swsearch.linkgraph.store",
    "swsearch.embed.paragraphs",
    "swsearch.metadata.store",
    "swsearch.index.faiss_store",
    "swsearch.mining.triplets",
    "swsearch.search.engine",
    "swsearch.eval.metrics",
    "swsearch.tools.inspect_index",
    "swsearch.fetch",
    "swsearch.pipeline",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)
