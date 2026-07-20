"""Install Delta Lake into the local pyspark so it works under pytest-xdist.

Why this exists
---------------
The Spark/Delta dataset tests used to be quarantined to the serial pytest pass. Moving them
to the parallel pass fails with `ClassNotFoundException: DeltaCatalog`, and the root cause is
not a race that a lock can fix -- it is the SparkSession singleton:

  * In local mode pyspark launches ONE JVM gateway per process. `spark.jars.packages` only
    adds jars to the classpath at that JVM's launch.
  * `SparkSession.builder...getOrCreate()` returns the process-global session if one already
    exists, silently ignoring the builder's `.config(...)`.
  * Under `--dist loadscope` a delta test module can land on a worker where a plain pyspark
    test (e.g. tests/pyspark/optuna) already launched a delta-less JVM. Our fixture then gets
    that session back, without delta on the classpath -> ClassNotFoundException.

The only robust fix is to make delta available *before the first JVM launch*, independent of
which test builds the first session. This script, run once per CI shard before pytest:

  1. Resolves the delta package once (serially, safely) and copies its jars into pyspark's
     own `jars/` dir so they are on the classpath at every JVM launch.
  2. Writes `spark-defaults.conf` enabling the delta SQL extension + catalog so every session
     in the process is delta-enabled, even one created by a non-delta test.

After this, delta works regardless of xdist worker/session ordering, so the tests can run in
the parallel pass.
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.data.spark_test_utils import delta_package


def main() -> None:
    import pyspark
    from pyspark.sql import SparkSession

    pyspark_dir = Path(pyspark.__file__).parent
    jars_dir = pyspark_dir / "jars"
    conf_dir = pyspark_dir / "conf"
    conf_dir.mkdir(exist_ok=True)

    # 1. Resolve the delta package once. This is the first session in a fresh, single-process
    #    setup step, so the Ivy resolution is safe and its jars land in the shared Ivy cache.
    with tempfile.TemporaryDirectory() as tmp:
        spark = (
            SparkSession.builder
            .master("local[1]")
            .config("spark.jars.packages", delta_package())
            .config("spark.sql.warehouse.dir", tmp)
            .getOrCreate()
        )
        spark.stop()

    # Copy every resolved package jar (delta-spark + transitive deps) onto pyspark's
    # classpath. `spark.jars.packages` retrieves them under ~/.ivy*/jars/.
    copied = 0
    for jar in Path.home().glob(".ivy*/jars/*.jar"):
        dest = jars_dir / jar.name
        if not dest.exists():
            shutil.copy(jar, dest)
            copied += 1
    if copied == 0:
        raise RuntimeError(
            "No delta jars were resolved into ~/.ivy*/jars; cannot set up delta on the classpath."
        )

    # 2. Enable the delta extension + catalog for every session in the process.
    (conf_dir / "spark-defaults.conf").write_text(
        "spark.sql.extensions io.delta.sql.DeltaSparkSessionExtension\n"
        "spark.sql.catalog.spark_catalog org.apache.spark.sql.delta.catalog.DeltaCatalog\n"
    )
    print(f"Installed delta ({delta_package()}): copied {copied} jars, wrote spark-defaults.conf")


if __name__ == "__main__":
    main()
