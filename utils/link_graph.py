import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list

def start_spark():
    return SparkSession.builder \
        .appName("LinkGraphCreator") \
        .master("local[*]") \
        .config("spark.driver.memory", "50g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .config("spark.local.dir", "../spark-temp") \
        .config("spark.driver.maxResultSize", "15g") \
        .getOrCreate()

def extract_sql_tuples(sc, file_path, pattern, extract_func):
    lines = sc.textFile(file_path)
    insert_lines = lines.filter(lambda l: l.startswith("INSERT INTO"))
    extracted = insert_lines.flatMap(lambda line: extract_func(pattern, line))
    return extracted

def parse_page(pattern, line):
    return [(int(pid), title.replace('_', ' ')) for pid, title in pattern.findall(line)]

def parse_pagelinks(pattern, line):
    return [(int(from_id), int(to_id)) for from_id, to_id in pattern.findall(line)]

def parse_linktarget(pattern, line):
    return [(int(lt_id), title.replace('_', ' ')) for lt_id, title in pattern.findall(line)]

def export_link_graph_to_jsonl(page_sql_path, pagelinks_sql_path, linktarget_sql_path, jsonl_output_path):
    spark = start_spark()
    sc = spark.sparkContext

    page_pattern = re.compile(r"\((\d+),\d+,'(.*?)',")
    pagelinks_pattern = re.compile(r"\((\d+),0,(\d+)\)")
    linktarget_pattern = re.compile(r"\((\d+),\d+,'(.*?)'\)")

    page_rdd = extract_sql_tuples(sc, page_sql_path, page_pattern, parse_page)
    pagelinks_rdd = extract_sql_tuples(sc, pagelinks_sql_path, pagelinks_pattern, parse_pagelinks)
    linktarget_rdd = extract_sql_tuples(sc, linktarget_sql_path, linktarget_pattern, parse_linktarget)

    page_df = spark.createDataFrame(page_rdd, ["page_id", "title"])
    pagelinks_df = spark.createDataFrame(pagelinks_rdd, ["from_id", "to_id"])
    linktarget_df = spark.createDataFrame(linktarget_rdd, ["lt_id", "lt_title"])

    joined_df = pagelinks_df \
        .join(page_df, pagelinks_df.from_id == page_df.page_id, "inner") \
        .join(linktarget_df, pagelinks_df.to_id == linktarget_df.lt_id, "inner") \
        .select(page_df.title.alias("from_title"), linktarget_df.lt_title.alias("to_title"))

    grouped = joined_df.groupBy("from_title").agg(collect_list("to_title").alias("linked_titles"))

    grouped.write.mode("overwrite").json(jsonl_output_path)

    spark.stop()


if __name__ == "__main__":
    page_sql_path = "../data/raw/enwiki-latest-page.sql"
    pagelinks_sql_path = "../data/raw/enwiki-latest-pagelinks.sql"
    linktarget_sql_path = "../data/raw/enwiki-latest-linktarget.sql"
    jsonl_output_path = "../data/processed/wiki_link_graph_jsonl"

    export_link_graph_to_jsonl(page_sql_path, pagelinks_sql_path, linktarget_sql_path, jsonl_output_path)
