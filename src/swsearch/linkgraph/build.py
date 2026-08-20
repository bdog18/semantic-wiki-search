import re

from pyspark.sql.functions import collect_list

from swsearch.common.spark import get_spark_session
from swsearch.logutil import get_logger

logger = get_logger(__name__)

_PAGE_PATTERN = re.compile(r"\((\d+),\d+,'(.*?)',")
_PAGELINKS_PATTERN = re.compile(r"\((\d+),0,(\d+)\)")
_LINKTARGET_PATTERN = re.compile(r"\((\d+),\d+,'(.*?)'\)")

# MySQL dump backslash-escapes (\', \", \\, ...), handled in one left-to-right
# pass so escaped-backslash-then-quote sequences can't be double-unescaped.
_MYSQL_ESCAPES = {
    "0": "\0",
    "'": "'",
    '"': '"',
    "b": "\b",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "Z": "\x1a",
    "\\": "\\",
}
_ESCAPE_RE = re.compile(r"\\(.)")


def unescape_mysql_string(s: str) -> str:
    """Undo MySQL dump string escaping. Fixes corruption at the source instead
    of patching it downstream when titles are later read back out of SQLite."""
    return _ESCAPE_RE.sub(lambda m: _MYSQL_ESCAPES.get(m.group(1), m.group(1)), s)


def extract_sql_tuples(sc, file_path, pattern, extract_func):
    lines = sc.textFile(file_path)
    insert_lines = lines.filter(lambda l: l.startswith("INSERT INTO"))
    return insert_lines.flatMap(lambda line: extract_func(pattern, line))


def parse_page(pattern, line):
    return [
        (int(pid), unescape_mysql_string(title).replace('_', ' '))
        for pid, title in pattern.findall(line)
    ]


def parse_pagelinks(pattern, line):
    return [(int(from_id), int(to_id)) for from_id, to_id in pattern.findall(line)]


def parse_linktarget(pattern, line):
    return [
        (int(lt_id), unescape_mysql_string(title).replace('_', ' '))
        for lt_id, title in pattern.findall(line)
    ]


def export_link_graph_to_jsonl(page_sql_path, pagelinks_sql_path, linktarget_sql_path, jsonl_output_path):
    spark = get_spark_session("LinkGraphCreator")
    try:
        _export(spark, page_sql_path, pagelinks_sql_path, linktarget_sql_path, jsonl_output_path)
    finally:
        # Matches embed/paragraphs.py: a stage that raises mid-job should
        # still release the session rather than leave a JVM behind.
        spark.stop()


def _export(spark, page_sql_path, pagelinks_sql_path, linktarget_sql_path, jsonl_output_path):
    sc = spark.sparkContext

    page_rdd = extract_sql_tuples(sc, page_sql_path, _PAGE_PATTERN, parse_page)
    pagelinks_rdd = extract_sql_tuples(sc, pagelinks_sql_path, _PAGELINKS_PATTERN, parse_pagelinks)
    linktarget_rdd = extract_sql_tuples(sc, linktarget_sql_path, _LINKTARGET_PATTERN, parse_linktarget)

    page_df = spark.createDataFrame(page_rdd, ["page_id", "title"])
    pagelinks_df = spark.createDataFrame(pagelinks_rdd, ["from_id", "to_id"])
    linktarget_df = spark.createDataFrame(linktarget_rdd, ["lt_id", "lt_title"])

    joined_df = (
        pagelinks_df.join(page_df, pagelinks_df.from_id == page_df.page_id, "inner")
        .join(linktarget_df, pagelinks_df.to_id == linktarget_df.lt_id, "inner")
        .select(page_df.title.alias("from_title"), linktarget_df.lt_title.alias("to_title"))
    )

    grouped = joined_df.groupBy("from_title").agg(collect_list("to_title").alias("linked_titles"))

    logger.info("Writing link graph JSONL to %s", jsonl_output_path)
    grouped.write.mode("overwrite").json(jsonl_output_path)
