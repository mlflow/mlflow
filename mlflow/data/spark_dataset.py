import json
import logging
import os
from functools import cached_property
from typing import Any, Dict, Optional, Union

import numpy as np
from pyspark.sql import SparkSession, DataFrame

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.digest_utils import get_normalized_md5_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
from mlflow.utils.uri import dbfs_hdfs_uri_to_fuse_path

_logger = logging.getLogger(__name__)


class SparkDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a Spark dataset (e.g. data derived from a Spark Table / file directory or Delta
    Table) for use with MLflow Tracking.
    """

    def __init__(
        self,
        df: DataFrame,
        source: DatasetSource,
        targets: Optional[str] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        if targets is not None and targets not in df.columns:
            raise MlflowException(
                f"The specified Spark dataset does not contain the specified targets column"
                f" '{targets}'.",
                INVALID_PARAMETER_VALUE,
            )

        self._df = df
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        # Retrieve a semantic hash of the DataFrame's logical plan, which is much more efficient
        # and deterministic than hashing DataFrame records
        return get_normalized_md5_digest([np.int64(self._df.semanticHash())])

    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        """
        :param base_dict: A string dictionary of base information about the
                          dataset, including: name, digest, source, and source
                          type.
        :return: A string dictionary containing the following fields: name,
                 digest, source, source type, schema (optional), profile
                 (optional).
        """
        base_dict.update(
            {
                "schema": json.dumps({"mlflow_colspec": self.schema.to_dict()}),
                "profile": json.dumps(self.profile),
            }
        )
        return base_dict

    @property
    def df(self) -> DataFrame:
        """
        The Spark DataFrame instance.

        :return: The Spark DataFrame instance.

        """
        return self._df

    @property
    def targets(self) -> Optional[str]:
        """
        The name of the Spark DataFrame column containing targets (labels) for supervised
        learning.

        :return: The string name of the Spark DataFrame column containing targets.
        """
        return self._targets

    @property
    def source(self) -> Union[SparkDatasetSource, DeltaDatasetSource]:
        """
        Spark dataset source information.

        :return: An instance of py:class:`SparkDatasetSource` or py:class:`DeltaDatasetSource`.
        """
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be None if no profile is available.
        """
        # use Spark RDD countApprox to get approximate count since count() may be expensive
        approx_count = self.df.rdd.countApprox(timeout=1000, confidence=0.90)

        return {
            "approx_count": approx_count,
        }

    @cached_property
    def schema(self) -> Optional[Schema]:
        """
        The MLflow ColSpec schema of the Spark dataset.
        """
        try:
            return _infer_schema(self._df)
        except Exception as e:
            _logger._warning("Failed to infer schema for Spark dataset. Exception: %s", e)
            return None

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the Spark DataFrame to pandas and splits the resulting
        `pandas.DataFrame` into: 1. a `pandas.DataFrame` of features and
        2. a `pandas.Series` of targets.

        To avoid overuse of driver memory, only the first 10,000 DataFrame rows are selected.
        """
        df = self._df.limit(10000).toPandas()
        if self._targets is not None:
            if self._targets not in df.columns:
                raise MlflowException(
                    f"Failed to convert Spark dataset to pyfunc inputs and outputs because"
                    f" the pandas representation of the Spark dataset does not contain the"
                    f" specified targets column '{self._targets}'.",
                    # This is an internal error because we should have validated the presence of
                    # the target column in the Hugging Face dataset at construction time
                    INTERNAL_ERROR,
                )
            inputs = df.drop(columns=self._targets)
            outputs = df[self._targets]
            return PyFuncInputsOutputs(inputs=inputs, outputs=outputs)
        else:
            return PyFuncInputsOutputs(inputs=df, outputs=None)


def load_delta(
    path: Optional[str] = None,
    table_name: Optional[str] = None,
    version: Optional[str] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> SparkDataset:
    """
    Loads a :py:class:`SparkDataset` from a Delta table for use with MLflow Tracking.

    :param path: The path to the Delta table. Either `path` or `table_name` must be specified.
    :param table_name: The name of the Delta table. Either `path` or `table_name` must be specified.
    :param version: The Delta table version. If not specified, the version will be inferred.
    :param targets: Optional. The name of the Delta table column containing targets (labels) for
                    supervised learning.
    :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                 automatically generated.
    :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                   is automatically computed.
    :return: An instance of :py:class:`SparkDataset`.
    """
    if (path, table_name).count(None) != 1:
        raise MlflowException(
            "Must specify exactly one of `table_name` or `path`.",
            INVALID_PARAMETER_VALUE,
        )

    if version is None:
        if path is not None:
            version = _try_get_delta_table_latest_version_from_path(path)
        else:
            version = _try_get_delta_table_latest_version_from_table_name(table_name)

    if name is None and table_name is not None:
        name = table_name + (f"v{version}" if version is not None else "")

    source = DeltaDatasetSource(path=path, delta_table_name=table_name, delta_table_version=version)
    df: DataFrame = source.load()

    return SparkDataset(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
    )


def from_spark(
    df: DataFrame,
    path: Optional[str] = None,
    table_name: Optional[str] = None,
    version: Optional[str] = None,
    sql: Optional[str] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> SparkDataset:
    """
    Given a Spark DataFrame, constructs an MLflow :py:class:`SparkDataset` object for use with
    MLflow Tracking.

    :param path: The path of the Spark or Delta source that the DataFrame originally came from.
                 Note that the path does not have to match the DataFrame exactly, since the
                 DataFrame may have been modified by Spark operations. This is used to reload the
                 dataset upon request via `SparkDataset.source.load()`. Either `path`,
                 `table_name`, or `sql` must be specified.
    :param table_name: The name of the Spark or Delta table that the DataFrame originally came from.
                       Note that the table does not have to match the DataFrame exactly, since the
                       DataFrame may have been modified by Spark operations. This is used to reload
                       the dataset upon request via `SparkDataset.source.load()`. Either `path`,
                       `table_name`, or `sql` must be specified.
    :param version: If the DataFrame originally came from a Delta table, specifies the version
                    of the Delta table. This is used to reload the dataset upon request via
                    `SparkDataset.source.load()`. `version` cannot be specified if `sql` is
                    specified.
    :param sql: The Spark SQL statement that was originally used to construct the DataFrame.
                Note that the Spark SQL statement does not have to match the DataFrame exactly,
                since the DataFrame may have been modified by Spark operations. This is used to
                reload the dataset upon request via `SparkDataset.source.load()`. Either `path`,
                `table_name`, or `sql` must be specified.
    :param targets: Optional. The name of the Data Frame column containing targets (labels) for
                    supervised learning.
    :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                 automatically generated.
    :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                   is automatically computed.
    :param df: A Spark DataFrame.
    :return: An instance of :py:class:`SparkDataset`.
    """
    if (path, table_name, sql).count(None) != 2:
        raise MlflowException(
            "Must specify exactly one of `path`, `table_name`, or `sql`.",
            INVALID_PARAMETER_VALUE,
        )

    if (sql, version).count(None) == 0:
        raise MlflowException(
            "`version` may not be specified when `sql` is specified. `version` may only be"
            " specified when `table_name` or `path` is specified.",
            INVALID_PARAMETER_VALUE,
        )

    if sql is not None:
        source = SparkDatasetSource(sql=sql)
    elif path is not None:
        if _is_delta_table_path(path):
            version = version or _try_get_delta_table_latest_version_from_path(path)
            source = DeltaDatasetSource(path=path, delta_table_version=version)
        elif version is None:
            source = SparkDatasetSource(path=path)
        else:
            raise MlflowException(
                f"Version '{version}' was specified, but the path '{path}' does not refer"
                f" to a Delta table.",
                INVALID_PARAMETER_VALUE,
            )
    elif table_name is not None:
        if _is_delta_table(table_name):
            version = version or _try_get_delta_table_latest_version_from_table_name(table_name)
            source = DeltaDatasetSource(
                delta_table_name=table_name,
                delta_table_version=version,
            )
        elif version is None:
            source = SparkDatasetSource(table_name=table_name)
        else:
            raise MlflowException(
                f"Version '{version}' was specified, but could not find a Delta table with name"
                f" '{table_name}'.",
                INVALID_PARAMETER_VALUE,
            )

    return SparkDataset(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
    )


def _is_delta_table(table_name: str) -> bool:
    """
    Checks if a Delta table exists with the specified table name.

    :return: True if a Delta table exists with the specified table name. False otherwise.
    """

    try:
        spark = SparkSession.builder.getOrCreate()
        table = spark.catalog.getTable(table_name)
        table_identifier = spark.sparkContext._jvm.org.apache.spark.sql.catalyst.TableIdentifier(
            table.name,
            spark.sparkContext._jvm.scala.Some(table.database),
            spark.sparkContext._jvm.scala.Some(table.catalog),
        )
        table_metadata = (
            spark._jsparkSession.sessionState().catalog().getTableMetadata(table_identifier)
        )
        table_provider = table_metadata.provider()
        return table_provider.isDefined() and table_provider.get() == "delta"
    except Exception:
        return False


def _is_delta_table_path(path: str) -> bool:
    """
    Checks if the specified filesystem path is a Delta table.

    :return: True if the specified path is a Delta table. False otherwise.
    """
    if os.path.exists(path) and "_delta_log" in os.listdir(path):
        return True

    try:
        dbfs_path = dbfs_hdfs_uri_to_fuse_path(path)
        return os.path.exists(dbfs_path) and "_delta_log" in os.listdir(dbfs_path)
    except Exception:
        return False


def _try_get_delta_table_latest_version_from_path(path: str) -> Optional[int]:
    """
    Gets the latest version of the Delta table located at the specified path.

    :param path: The path to the Delta table.
    :return: The version of the Delta table, or None if it cannot be resolved (e.g. because the
             Delta core library is not installed or the specified path does not refer to a Delta
             table).
    """
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


def _try_get_delta_table_latest_version_from_table_name(table_name: str) -> Optional[int]:
    """
    Gets the latest version of the Delta table with the specified name.

    :param table_name: The name of the Delta table.
    :return: The version of the Delta table, or None if it cannot be resolved (e.g. because the
             Delta core library is not installed or no such table exists).
    """
    try:
        spark = SparkSession.builder.getOrCreate()
        j_delta_table = spark._jvm.io.delta.tables.DeltaTable.forName(
            spark._jsparkSession, table_name
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
    """
    Obtains the latest version of the specified Delta table Java class.

    :param delta_table: A Java DeltaTable class instance.
    :return: The version of the Delta table.
    """
    latest_commit_jdf = j_delta_table.history(1)
    latest_commit_row = latest_commit_jdf.head()
    version_field_idx = latest_commit_row.fieldIndex("version")
    return latest_commit_row.get(version_field_idx)
