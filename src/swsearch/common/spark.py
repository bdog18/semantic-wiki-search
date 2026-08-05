import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession

from swsearch.config import REPO_ROOT, settings

# Pin worker/driver Python to sys.executable (the current venv) rather than
# letting PySpark fall back to whatever `python3` is first on PATH -- without
# this, local[*] workers spawn using the system Python, which has neither
# swsearch nor its dependencies installed. Any job that ships a closure
# referencing swsearch.* (e.g. linkgraph/build.py's SQL-line parsers, pickled
# and sent to workers by flatMap) then fails on the worker side with
# `ModuleNotFoundError: No module named 'swsearch'`. This has to be an
# environment variable, set before the JVM/SparkContext is created --
# `spark.pyspark.python`/`spark.pyspark.driver.python` SparkConf keys exist
# but were not honored for local-mode worker spawning in testing here.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def get_spark_session(
    app_name: str,
    driver_memory: str | None = None,
    max_result_size: str | None = None,
) -> SparkSession:
    """Single canonical Spark session builder.

    Replaces the three drifted ad-hoc SparkSession.builder configs that used
    to live in link_graph.py, page_links.py, and spark_functions.py.
    """
    s = settings.spark

    local_dir = Path(s.local_dir)
    if not local_dir.is_absolute():
        local_dir = REPO_ROOT / local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", driver_memory or s.driver_memory)
        .config("spark.sql.shuffle.partitions", str(s.shuffle_partitions))
        .config("spark.local.dir", str(local_dir))
        .config("spark.driver.maxResultSize", max_result_size or s.max_result_size)
        .getOrCreate()
    )
