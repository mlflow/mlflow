"""Resolve the Spark/Delta jars once into the shared Ivy cache before the parallel test pass.

The Spark/Delta dataset tests (tests/data/test_spark_dataset*.py, test_delta_dataset_source.py,
test_pandas_dataset.py) run under pytest-xdist. Each SparkSession pulls the delta package via
`spark.jars.packages`, which triggers Ivy resolution into the shared ~/.ivy2 cache. Ivy's
resolver is not concurrency-safe, so several xdist workers cold-resolving the same uncached
package simultaneously corrupt each other and fail with `ClassNotFoundException: DeltaCatalog`.

Running one SparkSession here, serially, resolves and caches the jar so every worker later hits
a warm cache. Concurrent cache *reads* are safe; only concurrent *resolution* is not.

The package version must match what the test fixtures request.
"""

from packaging.version import Version


def main() -> None:
    import pyspark
    from pyspark.sql import SparkSession

    # Mirror the version selection in the test fixtures.
    delta_package = (
        "io.delta:delta-spark_2.13:4.0.0"
        if Version(pyspark.__version__).major >= 4
        else "io.delta:delta-spark_2.12:3.0.0"
    )

    spark = (
        SparkSession.builder
        .master("local[1]")
        .config("spark.jars.packages", delta_package)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .getOrCreate()
    )
    # Touch the Delta catalog so resolution actually happens (not just a lazy config).
    spark.sql("SELECT 1").collect()
    spark.stop()
    print(f"Pre-warmed Ivy cache for {delta_package}")


if __name__ == "__main__":
    main()
