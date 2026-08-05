"""Fetch raw Wikipedia dump inputs if they aren't already on disk.

Ports notebooks/archive/main.ipynb's Steps 1-5 (install wikiextractor, download
the XML dump, extract it, download the 3 SQL link-graph dumps) into idempotent
Python functions: each one checks for its expected output first and does
nothing if it's already there, so a completed download or extraction is never
redone. URLs are the same dumps.wikimedia.org / GitHub sources the notebook
already used.
"""
import gzip
import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests

from swsearch.config import REPO_ROOT, settings
from swsearch.logutil import get_logger

logger = get_logger(__name__)

WIKIEXTRACTOR_ZIP_URL = "https://github.com/attardi/wikiextractor/archive/refs/heads/master.zip"
XML_DUMP_URL = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
SQL_DUMP_URLS = {
    "page": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-page.sql.gz",
    "pagelinks": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pagelinks.sql.gz",
    "linktarget": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-linktarget.sql.gz",
}


def _download(url: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
    """Stream-download url to dest_path via a .part temp file, so a killed
    download can't be mistaken for a complete one on the next run."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_name(dest_path.name + ".part")
    logger.info("Downloading %s -> %s", url, dest_path)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        written = 0
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk)
    tmp_path.rename(dest_path)
    logger.info("Downloaded %s (%d bytes)", dest_path, written)


def _is_wikiextractor_installed() -> bool:
    return importlib.util.find_spec("wikiextractor") is not None


def ensure_wikiextractor_installed() -> None:
    """Download (if needed) and pip-install the vendored wikiextractor tool."""
    if _is_wikiextractor_installed():
        return

    wikiextractor_dir = REPO_ROOT / "wikiextractor-master"
    if not wikiextractor_dir.is_dir():
        logger.info("wikiextractor source not found; downloading from GitHub...")
        zip_path = REPO_ROOT / "wikiextractor.zip"
        _download(WIKIEXTRACTOR_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(REPO_ROOT)
        zip_path.unlink()

    logger.info("Installing wikiextractor from %s...", wikiextractor_dir)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(wikiextractor_dir)], check=True)


def ensure_xml_dump() -> Path:
    dest_path = settings.paths.raw_dir / "enwiki-latest-pages-articles.xml.bz2"
    if dest_path.is_file():
        logger.info("Wikipedia XML dump already present at %s", dest_path)
    else:
        _download(XML_DUMP_URL, dest_path)
    return dest_path


def ensure_sql_dumps() -> tuple[Path, Path, Path]:
    """Download and decompress the 3 SQL link-graph dumps if missing. Returns
    (page_sql_path, pagelinks_sql_path, linktarget_sql_path)."""
    paths = settings.paths
    targets = {
        "page": paths.page_sql_path,
        "pagelinks": paths.pagelinks_sql_path,
        "linktarget": paths.linktarget_sql_path,
    }
    for name, dest in targets.items():
        if dest.is_file():
            logger.info("%s.sql already present at %s", name, dest)
            continue
        gz_path = dest.with_name(dest.name + ".gz")
        _download(SQL_DUMP_URLS[name], gz_path)
        logger.info("Decompressing %s -> %s", gz_path, dest)
        tmp_dest = dest.with_name(dest.name + ".part")
        with gzip.open(gz_path, "rb") as f_in, open(tmp_dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        tmp_dest.rename(dest)
        gz_path.unlink()
    return targets["page"], targets["pagelinks"], targets["linktarget"]


def ensure_wikiextractor_output() -> Path:
    """Run wikiextractor over the XML dump if data/raw/extracted_wikidata
    doesn't already exist (and isn't empty)."""
    extracted_dir = settings.paths.extracted_dir
    if extracted_dir.is_dir() and any(extracted_dir.iterdir()):
        logger.info("Extracted wikidata already present at %s", extracted_dir)
        return extracted_dir

    ensure_wikiextractor_installed()
    xml_dump_path = ensure_xml_dump()
    extracted_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running wikiextractor on %s -> %s (this can take a while)...", xml_dump_path, extracted_dir)
    subprocess.run(
        [
            sys.executable, "-m", "wikiextractor.WikiExtractor",
            str(xml_dump_path), "-o", str(extracted_dir), "--no-templates",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
    return extracted_dir


def ensure_raw_data() -> None:
    """Ensure data/raw/extracted_wikidata and the 3 SQL link-graph dumps
    exist, downloading/extracting whatever's missing. Fully idempotent --
    anything already present on disk is left untouched, not re-fetched."""
    ensure_wikiextractor_output()
    ensure_sql_dumps()
