from unittest.mock import MagicMock

import pytest

import swsearch.embed.paragraphs as embed_paragraphs_mod
import swsearch.extract.wikidump as wikidump_mod
import swsearch.fetch as fetch_mod
import swsearch.index.faiss_store as faiss_store_mod
import swsearch.linkgraph.backlinks as linkgraph_backlinks_mod
import swsearch.linkgraph.build as linkgraph_build_mod
import swsearch.linkgraph.store as linkgraph_store_mod
import swsearch.metadata.store as metadata_store_mod
import swsearch.mining.triplets as triplets_mod
from swsearch import pipeline


@pytest.fixture
def mocked_stages(monkeypatch):
    calls = []

    def record(name):
        def _fn(*args, **kwargs):
            calls.append(name)
            return MagicMock()
        return _fn

    monkeypatch.setattr(fetch_mod, "ensure_raw_data", record("fetch"))
    monkeypatch.setattr(linkgraph_build_mod, "export_link_graph_to_jsonl", record("build_linkgraph.export"))
    monkeypatch.setattr(linkgraph_store_mod, "build_linkgraph_sqlite", record("build_linkgraph.sqlite"))
    monkeypatch.setattr(linkgraph_backlinks_mod, "build_backlink_counts_sqlite", record("build_backlinks"))
    monkeypatch.setattr(wikidump_mod, "traverse_directory", record("extract.traverse"))
    monkeypatch.setattr(wikidump_mod, "convert_json_array_to_jsonl", record("extract.convert"))
    monkeypatch.setattr(wikidump_mod, "save_article_titles", record("extract.titles"))
    monkeypatch.setattr(embed_paragraphs_mod, "embed_paragraphs", record("embed"))
    monkeypatch.setattr(metadata_store_mod, "load_faiss_meta_sqlite", lambda path: MagicMock())
    # The pipeline calls the one entry point that owns the flat-vs-ivfpq
    # decision; the type it was asked for is recorded alongside the call.
    def record_build_index(embeddings_dir, meta_conn, index_path, index_type="auto"):
        calls.append(f"build_index[{index_type}]")
        return MagicMock()

    monkeypatch.setattr(faiss_store_mod, "build_index_from_manifest", record_build_index)
    monkeypatch.setattr(triplets_mod, "mine_triplets", record("mine_triplets"))

    return calls


def test_pipeline_runs_all_stages_in_order_by_default(mocked_stages):
    pipeline.run_full_pipeline()

    assert mocked_stages == [
        "fetch",
        "build_linkgraph.export",
        "build_linkgraph.sqlite",
        # Reranking is the biggest quality lever in the system and a missing
        # backlink db degrades silently, so the pipeline must produce it.
        "build_backlinks",
        "extract.traverse",
        "extract.convert",
        "extract.titles",
        "embed",
        "build_index[auto]",
    ]


def test_pipeline_skip_fetch(mocked_stages):
    pipeline.run_full_pipeline(skip_fetch=True)

    assert "fetch" not in mocked_stages
    # everything downstream still ran
    assert "build_linkgraph.export" in mocked_stages
    assert "build_backlinks" in mocked_stages
    assert "build_index[auto]" in mocked_stages


def test_pipeline_with_triplets_appends_mining_stage(mocked_stages):
    pipeline.run_full_pipeline(with_triplets=True)

    assert mocked_stages[-1] == "mine_triplets"


def test_pipeline_without_triplets_by_default(mocked_stages):
    pipeline.run_full_pipeline()

    assert "mine_triplets" not in mocked_stages


def test_pipeline_passes_explicit_index_type_through(mocked_stages):
    pipeline.run_full_pipeline(index_type="ivfpq")

    assert "build_index[ivfpq]" in mocked_stages


def test_pipeline_defaults_to_auto_index_type(mocked_stages):
    pipeline.run_full_pipeline()

    assert "build_index[auto]" in mocked_stages


def test_pipeline_rejects_bad_index_type_before_doing_any_work(mocked_stages):
    # Validated up front: a typo should cost a second, not a day of embedding.
    with pytest.raises(ValueError):
        pipeline.run_full_pipeline(index_type="bogus")

    assert mocked_stages == []


def test_format_elapsed():
    assert pipeline._format_elapsed(5) == "5s"
    assert pipeline._format_elapsed(65) == "1m05s"
    assert pipeline._format_elapsed(3665) == "1h01m05s"
