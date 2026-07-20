"""Shared Spark session builder for the data tests, safe under pytest-xdist.

These modules used to be quarantined to the serial pytest pass. The blocker was the
SparkSession singleton: in local mode pyspark launches one JVM per process, `spark.jars.packages`
only adds jars at that JVM's launch, and `getOrCreate()` returns any pre-existing session while
silently ignoring the builder's `.config(...)`. Under `--dist loadscope` a delta test could land
on a worker where a plain pyspark test already launched a delta-less JVM, so the fixture got a
session without delta on the classpath -> `ClassNotFoundException: DeltaCatalog`.

The fix lives in CI setup (dev/setup_delta_spark.py): delta is installed onto pyspark's own
classpath and enabled via spark-defaults.conf *before any test runs*, so every session in the
process is delta-enabled regardless of which test builds the first one. That removes the need
for `spark.jars.packages` here. This builder only isolates the Derby metastore + warehouse to a
per-module tmp dir so parallel workers don't collide on the default ./metastore_db.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from packaging.version import Version

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def delta_package() -> str:
    """The delta-spark Maven coordinate matching the installed pyspark major version.

    Used by dev/setup_delta_spark.py to resolve the jars; kept here as the single source of
    truth for the coordinate.
    """
    import pyspark

    if Version(pyspark.__version__).major >= 4:
        return "io.delta:delta-spark_2.13:4.0.0"
    return "io.delta:delta-spark_2.12:3.0.0"


def build_spark_session(tmp_dir: Path) -> "SparkSession":
    """Build a delta-enabled local SparkSession with an isolated metastore.

    Delta is already on the classpath + enabled via spark-defaults.conf (see
    dev/setup_delta_spark.py), so no `spark.jars.packages` is needed here. `tmp_dir` should be
    a per-module temp dir (e.g. from tmp_path_factory) so the Derby metastore and warehouse are
    isolated from sibling xdist workers, which otherwise collide on the default ./metastore_db.
    """
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder
        .master("local[*]")
        .config("spark.sql.warehouse.dir", str(tmp_dir))
        .config(
            "javax.jdo.option.ConnectionURL",
            f"jdbc:derby:;databaseName={tmp_dir}/metastore_db;create=true",
        )
        .getOrCreate()
    )
