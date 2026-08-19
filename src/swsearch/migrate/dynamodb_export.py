"""Export the FAISS metadata SQLite store to DynamoDB-JSON for bulk import.

Writes gzipped, newline-delimited DynamoDB-JSON that `aws dynamodb
import-table` ingests directly from S3. That path costs roughly $0.15 per
uncompressed GB -- about $3 for this 21GB store -- against roughly $50 of
write units to push 41.95M items through BatchWriteItem, and it parallelises
itself instead of running for hours in a loop here.

Three transformations happen on the way out, all of them things that are
free during a pass we have to make anyway and expensive to do later:

  1. Markup paragraphs are dropped (extract.wikitext.clean_paragraph). A ~1%
     contamination rate in returned results comes from ~0.05% of rows; see
     that module for the measurements.
  2. The article's curid is denormalised onto every paragraph, joined from
     article_titles.json. That retires a 300MB JSON file the engine
     currently loads into a ~1GB in-memory dict purely to resolve a URL it
     could have carried alongside the text.
  3. idx stays the partition key, so lookups remain the same integer point
     reads the SQLite store served.

Dropping rows is safe: search.engine skips candidates whose metadata is
missing, and its candidate pool (max(k*10, 50)) has slack for it. Note this
does break faiss_store._assert_aligned's index.ntotal == row count
invariant, which is a build-time check and does not run during serving.

Usage:
    python -m swsearch.migrate.dynamodb_export --out-dir /path/to/export
    aws s3 cp --recursive /path/to/export s3://BUCKET/metadata-import/
"""

import argparse
import gzip
import json
import re
import sqlite3
import sys
from pathlib import Path

from swsearch.config import settings
from swsearch.extract.wikitext import clean_paragraph
from swsearch.logutil import get_logger

logger = get_logger(__name__)

# article_titles.json stores full URLs (".../wiki?curid=33637105"); only the
# id is worth carrying, and as an int it costs a few bytes per item instead
# of a ~46-character string.
_CURID_RE = re.compile(r"curid=(\d+)")

# Rows per output file. DynamoDB's import parallelises across objects, so a
# single enormous file would serialise the load; ~1M rows lands around
# 150-250MB gzipped, comfortably inside the 5GB per-object import limit.
DEFAULT_ROWS_PER_FILE = 1_000_000


def load_curid_map(path: Path) -> dict[str, int]:
    """title -> curid, parsed out of article_titles.json's URL values."""
    logger.info("Loading article titles from %s", path)
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)

    curids: dict[str, int] = {}
    for title, url in raw.items():
        match = _CURID_RE.search(url or "")
        if match:
            curids[title] = int(match.group(1))
    logger.info("Resolved %d/%d titles to a curid", len(curids), len(raw))
    return curids


def _item(idx: int, text: str, title: str, curid: int | None) -> str:
    """One line of DynamoDB-JSON. Attribute names are unquoted keys here;
    `text` is only a reserved word inside expressions, not in item payloads.
    """
    attributes = {
        "idx": {"N": str(idx)},
        "text": {"S": text},
        "title": {"S": title},
    }
    if curid is not None:
        attributes["curid"] = {"N": str(curid)}
    return json.dumps({"Item": attributes}, ensure_ascii=False)


def export(meta_db_path: Path, titles_path: Path, out_dir: Path, rows_per_file: int, min_chars: int, limit: int | None = None) -> None:
    curids = load_curid_map(titles_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{meta_db_path}?mode=ro", uri=True)
    total = connection.execute("SELECT COUNT(*) FROM faiss_meta").fetchone()[0]
    logger.info("Exporting %d rows from %s", total, meta_db_path)

    read, written, dropped, cleaned, missing_curid = 0, 0, 0, 0, 0
    handle, file_index = None, 0

    # Streamed, not fetchall(): the store is 21GB and materialising it would
    # need more memory than any sensible machine has.
    cursor = connection.execute("SELECT idx, text, article_title FROM faiss_meta ORDER BY idx")
    for idx, text, title in cursor:
        if limit is not None and read >= limit:
            break
        read += 1

        prose = clean_paragraph(text, min_chars=min_chars)
        if prose is None:
            dropped += 1
            continue
        if prose != text:
            cleaned += 1

        curid = curids.get(title)
        if curid is None:
            missing_curid += 1

        if handle is None or written % rows_per_file == 0:
            if handle is not None:
                handle.close()
            path = out_dir / f"faiss_meta_{file_index:04d}.jsonl.gz"
            handle = gzip.open(path, "wt", encoding="utf-8")
            logger.info("Writing %s", path)
            file_index += 1

        handle.write(_item(idx, prose, title, curid) + "\n")
        written += 1

        if read % 1_000_000 == 0:
            logger.info("  %d/%d read, %d written, %d dropped", read, total, written, dropped)

    if handle is not None:
        handle.close()

    logger.info(
        "Done: %d read, %d written (%d files), %d dropped as markup, "
        "%d had inline markup stripped, %d had no curid",
        read, written, file_index, dropped, cleaned, missing_curid,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--meta-db", type=Path, default=settings.paths.faiss_meta_db_path)
    parser.add_argument("--titles", type=Path, default=settings.paths.article_titles_path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--rows-per-file", type=int, default=DEFAULT_ROWS_PER_FILE)
    parser.add_argument(
        "--min-chars",
        type=int,
        default=0,
        help="Drop paragraphs shorter than this after cleaning. Defaults to 0 "
             "(keep whatever was indexed) -- extraction already applied its own "
             "threshold, and dropping more here only widens the gap between the "
             "index and its metadata.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N rows (for a trial run).")
    args = parser.parse_args(argv)

    if args.limit is not None:
        logger.warning("--limit %d: this is a partial export, not a full migration", args.limit)

    export(args.meta_db, args.titles, args.out_dir, args.rows_per_file, args.min_chars, args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
