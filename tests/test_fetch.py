import gzip
from unittest.mock import MagicMock

import pytest

from swsearch import fetch
from swsearch.config import PathSettings, Settings


@pytest.fixture
def fake_settings(tmp_path, monkeypatch):
    fake = Settings(paths=PathSettings(data_root=tmp_path))
    monkeypatch.setattr(fetch, "settings", fake)
    return fake


def test_download_sends_a_descriptive_user_agent(tmp_path, monkeypatch):
    # dumps.wikimedia.org 403s requests' default "python-requests/x.y" UA --
    # confirmed live against the real endpoint, identical request otherwise.
    captured = {}

    class FakeResponse:
        headers = {}

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            return iter([b"data"])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_get(url, stream, timeout, headers):
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(fetch.requests, "get", fake_get)

    fetch._download("https://example.org/f", tmp_path / "f")

    assert "User-Agent" in captured["headers"]
    assert captured["headers"]["User-Agent"] != ""
    assert "python-requests" not in captured["headers"]["User-Agent"]


def test_ensure_xml_dump_skips_when_present(fake_settings, monkeypatch):
    dest = fake_settings.paths.raw_dir / "enwiki-latest-pages-articles.xml.bz2"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"already here")

    download = MagicMock()
    monkeypatch.setattr(fetch, "_download", download)

    result = fetch.ensure_xml_dump()

    assert result == dest
    assert dest.read_bytes() == b"already here"
    download.assert_not_called()


def test_ensure_xml_dump_downloads_when_missing(fake_settings, monkeypatch):
    download = MagicMock()
    monkeypatch.setattr(fetch, "_download", download)

    dest = fake_settings.paths.raw_dir / "enwiki-latest-pages-articles.xml.bz2"
    result = fetch.ensure_xml_dump()

    assert result == dest
    download.assert_called_once_with(fetch.XML_DUMP_URL, dest)


def test_download_uses_a_process_unique_temp_filename(tmp_path):
    # Two overlapping downloads to the same dest must never share a temp
    # path -- this is what let one process's completed rename crash a
    # concurrent process's rename of an already-renamed-away .part file.
    tmp_1 = fetch._unique_temp_path(tmp_path / "f")
    tmp_2 = fetch._unique_temp_path(tmp_path / "f")

    assert tmp_1 != tmp_2
    assert tmp_1.parent == tmp_path
    assert tmp_1.name.startswith("f.part-")


def test_download_cleans_up_temp_file_on_partial_failure(tmp_path, monkeypatch):
    class FlakyResponse:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield b"partial data"
            raise ConnectionError("connection dropped mid-stream")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: FlakyResponse())

    with pytest.raises(ConnectionError):
        fetch._download("https://example.org/f", tmp_path / "f")

    assert list(tmp_path.glob("f.part-*")) == []  # partial temp file was cleaned up, not left as debris


def test_ensure_sql_dumps_skips_present_and_fetches_missing(fake_settings, monkeypatch):
    # page.sql is already there; pagelinks/linktarget are not.
    fake_settings.paths.page_sql_path.parent.mkdir(parents=True, exist_ok=True)
    fake_settings.paths.page_sql_path.write_bytes(b"already here")

    fetched_urls = {}

    def fake_download(url, dest_path):
        fetched_urls[dest_path] = url
        with gzip.open(dest_path, "wb") as f:
            f.write(b"sql content")

    monkeypatch.setattr(fetch, "_download", fake_download)

    page, pagelinks, linktarget = fetch.ensure_sql_dumps()

    assert page == fake_settings.paths.page_sql_path
    assert page.read_bytes() == b"already here"  # untouched, not redownloaded
    assert pagelinks.read_bytes() == b"sql content"
    assert linktarget.read_bytes() == b"sql content"
    assert len(fetched_urls) == 2  # only the two missing dumps were fetched
    assert not pagelinks.with_name(pagelinks.name + ".gz").exists()  # .gz cleaned up


def test_ensure_sql_dumps_skips_download_when_gz_already_present(fake_settings, monkeypatch):
    # e.g. a prior run downloaded page.sql.gz fully but crashed before
    # decompressing it -- a retry shouldn't re-download ~2GB for nothing.
    gz_path = fake_settings.paths.page_sql_path.with_name(fake_settings.paths.page_sql_path.name + ".gz")
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_path, "wb") as f:
        f.write(b"already downloaded")

    fetched = []

    def fake_download(url, dest_path):
        fetched.append(dest_path)
        with gzip.open(dest_path, "wb") as f:
            f.write(b"freshly downloaded")

    monkeypatch.setattr(fetch, "_download", fake_download)

    page, pagelinks, linktarget = fetch.ensure_sql_dumps()

    assert page.read_bytes() == b"already downloaded"
    assert not gz_path.exists()  # decompressed then cleaned up
    assert pagelinks.read_bytes() == b"freshly downloaded"
    assert linktarget.read_bytes() == b"freshly downloaded"
    # page's .gz was already there -- only pagelinks/linktarget got downloaded
    assert fetched == [
        pagelinks.with_name(pagelinks.name + ".gz"),
        linktarget.with_name(linktarget.name + ".gz"),
    ]


def test_ensure_wikiextractor_output_skips_when_present(fake_settings, monkeypatch):
    extracted = fake_settings.paths.extracted_dir
    extracted.mkdir(parents=True)
    (extracted / "AA").mkdir()

    run = MagicMock()
    monkeypatch.setattr(fetch.subprocess, "run", run)

    result = fetch.ensure_wikiextractor_output()

    assert result == extracted
    run.assert_not_called()


def test_ensure_wikiextractor_installed_skips_when_already_importable(monkeypatch):
    monkeypatch.setattr(fetch, "_is_wikiextractor_installed", lambda: True)
    run = MagicMock()
    monkeypatch.setattr(fetch.subprocess, "run", run)

    fetch.ensure_wikiextractor_installed()

    run.assert_not_called()


def test_ensure_wikiextractor_installed_installs_from_existing_source(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch, "_is_wikiextractor_installed", lambda: False)
    monkeypatch.setattr(fetch, "REPO_ROOT", tmp_path)
    (tmp_path / "wikiextractor-master").mkdir()

    download = MagicMock()
    monkeypatch.setattr(fetch, "_download", download)
    run = MagicMock()
    monkeypatch.setattr(fetch.subprocess, "run", run)

    fetch.ensure_wikiextractor_installed()

    download.assert_not_called()  # source already present, no need to fetch the zip
    run.assert_called_once()
    args = run.call_args[0][0]
    assert "-e" in args and str(tmp_path / "wikiextractor-master") in args
