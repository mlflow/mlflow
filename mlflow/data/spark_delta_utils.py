import logging
import os

from mlflow.utils.string_utils import _backtick_quote

_logger = logging.getLogger(__name__)


def _is_delta_table(table_name: str) -> bool:
    """Checks if a Delta table exists with the specified table name.

    Returns:
        True if a Delta table exists with the specified table name. False otherwise.

    """
    from pyspark.sql import SparkSession
    from pyspark.sql.utils import AnalysisException

    spark = SparkSession.builder.getOrCreate()

    try:
        # use DESCRIBE DETAIL to check if the table is a Delta table
        # https://docs.databricks.com/delta/delta-utility.html#describe-detail
        # format will be `delta` for delta tables
        spark.sql(f"DESCRIBE DETAIL {table_name}").filter("format = 'delta'").count()
        return True
    except AnalysisException:
        return False


def _is_delta_table_path(path: str) -> bool:
    """Checks if the specified filesystem path is a Delta table.

    Returns:
        True if the specified path is a Delta table. False otherwise.
    """
    if os.path.exists(path) and os.path.isdir(path) and "_delta_log" in os.listdir(path):
        return True
    from mlflow.utils.uri import dbfs_hdfs_uri_to_fuse_path

    try:
        dbfs_path = dbfs_hdfs_uri_to_fuse_path(path)
        return os.path.exists(dbfs_path) and "_delta_log" in os.listdir(dbfs_path)
    except Exception:
        return False


def _try_get_delta_table_latest_version_from_path(path: str) -> int | None:
    """Gets the latest version of the Delta table located at the specified path.

    Args:
        path: The path to the Delta table.

    Returns:
        The version of the Delta table, or None if it cannot be resolved (e.g. because the
        Delta core library is not installed or the specified path does not refer to a Delta
        table).

    """
    from pyspark.sql import SparkSession

    try:
        spark = SparkSession.builder.getOrCreate()
        j_delta_table = spark._jvm.io.delta.tables.DeltaTable.forPath(spark._jsparkSession, path)
        return _get_delta_table_latest_version(j_delta_table)
    except Exception as e:
        _logger.warning(
            "Failed to obtain version information for Delta table at path '%s'. Version information"
            " may not be included in the dataset source for MLflow Tracking. Exception: %s",
            path,
            e,
        )


def _try_get_delta_table_latest_version_from_table_name(table_name: str) -> int | None:
    """Gets the latest version of the Delta table with the specified name.

    Args:
        table_name: The name of the Delta table.

    Returns:
        The version of the Delta table, or None if it cannot be resolved (e.g. because the
        Delta core library is not installed or no such table exists).
    """
    from pyspark.sql import SparkSession

    try:
        spark = SparkSession.builder.getOrCreate()
        backticked_table_name = ".".join(map(_backtick_quote, table_name.split(".")))
        j_delta_table = spark._jvm.io.delta.tables.DeltaTable.forName(
            spark._jsparkSession, backticked_table_name
        )
        return _get_delta_table_latest_version(j_delta_table)
    except Exception as e:
        _logger.warning(
            "Failed to obtain version information for Delta table with name '%s'. Version"
            " information may not be included in the dataset source for MLflow Tracking."
            " Exception: %s",
            table_name,
            e,
        )


def _get_delta_table_latest_version(j_delta_table) -> int:
    """Obtains the latest version of the specified Delta table Java class.

    Args:
        j_delta_table: A Java DeltaTable class instance.

    Returns:
        The version of the Delta table.

    """
    latest_commit_jdf = j_delta_table.history(1)
    latest_commit_row = latest_commit_jdf.head()
    version_field_idx = latest_commit_row.fieldIndex("version")
    return latest_commit_row.get(version_field_idx)
