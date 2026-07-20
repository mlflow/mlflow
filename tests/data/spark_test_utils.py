"""Shared Spark session builder for the data tests, safe under pytest-xdist.

These modules used to be quarantined to the serial pytest pass. The blocker is Ivy: each
SparkSession created with `spark.jars.packages` resolves the delta jar through Ivy, whose
resolve/retrieve step is not concurrency-safe. When several xdist workers call getOrCreate()
at once they race on the Ivy locks and some sessions come up with an incomplete classpath,
failing later delta writes with `ClassNotFoundException: DeltaCatalog` (no error at build
time). Neither a per-worker cache (4 cold resolutions racing) nor a pre-warmed shared cache
(still a lock race per session) fixes it.

The fix is to serialize just the session build with a cross-process file lock: the Ivy step
takes a few seconds and only needs to happen safely once per worker; tests then run fully in
parallel. The Derby metastore and warehouse are pinned to a per-module tmp dir so workers
don't collide on the default ./metastore_db either.
"""

import tempfile
from pathlib import Path

from filelock import FileLock
from packaging.version import Version

# A fixed lock path shared across all xdist workers on the host. Only one worker resolves
# Spark/Delta jars through Ivy at a time; the lock is released as soon as the session is up.
_SPARK_BUILD_LOCK = Path(tempfile.gettempdir()) / "mlflow_spark_session_build.lock"


def delta_package() -> str:
    """The delta-spark Maven coordinate matching the installed pyspark major version."""
    import pyspark

    if Version(pyspark.__version__).major >= 4:
        return "io.delta:delta-spark_2.13:4.0.0"
    return "io.delta:delta-spark_2.12:3.0.0"


def build_spark_session(tmp_dir: Path):
    """Build a delta-enabled local SparkSession, serializing the Ivy resolution step.

    `tmp_dir` should be a per-module temp dir (e.g. from tmp_path_factory) so the Derby
    metastore and warehouse are isolated from sibling xdist workers.
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .master("local[*]")
        .config("spark.jars.packages", delta_package())
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.warehouse.dir", str(tmp_dir))
        .config(
            "javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={tmp_dir}/metastore_db;create=true",
        )
    )
    with FileLock(str(_SPARK_BUILD_LOCK)):
        return builder.getOrCreate()
