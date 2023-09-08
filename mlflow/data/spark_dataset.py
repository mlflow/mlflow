import json
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.digest_utils import compute_spark_df_digest
from mlflow.data.pyfunc_dataset_mixin import PyFuncConvertibleDatasetMixin, PyFuncInputsOutputs
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.models.evaluation.base import EvaluationDataset
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, INVALID_PARAMETER_VALUE
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema
from mlflow.utils.annotations import experimental

if TYPE_CHECKING:
    import pyspark

_logger = logging.getLogger(__name__)


@experimental
class SparkDataset(Dataset, PyFuncConvertibleDatasetMixin):
    """
    Represents a Spark dataset (e.g. data derived from a Spark Table / file directory or Delta
    Table) for use with MLflow Tracking.
    """

    def __init__(
        self,
        df: "pyspark.sql.DataFrame",
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
        self._targets = targets
        super().__init__(source=source, name=name, digest=digest)

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        # Retrieve a semantic hash of the DataFrame's logical plan, which is much more efficient
        # and deterministic than hashing DataFrame records
        return compute_spark_df_digest(self._df)

    def _to_dict(self, base_dict: Dict[str, str]) -> Dict[str, str]:
        """
        :param base_dict: A string dictionary of base information about the
                          dataset, including: name, digest, source, and source
                          type.
        :return: A string dictionary containing the following fields: name,
                 digest, source, source type, schema (optional), profile
                 (optional).
        """
        return {
            **base_dict,
            "schema": json.dumps({"mlflow_colspec": self.schema.to_dict()})
            if self.schema
            else None,
            "profile": json.dumps(self.profile),
        }

    @property
    def df(self):
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

        :return: An instance of :py:class:`SparkDatasetSource
            <mlflow.data.spark_dataset_source.SparkDatasetSource>`
            or :py:class:`DeltaDatasetSource
            <mlflow.data.delta_dataset_source.DeltaDatasetSource>`.
        """
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        """
        A profile of the dataset. May be None if no profile is available.
        """
        try:
            from pyspark.rdd import BoundedFloat

            # Use Spark RDD countApprox to get approximate count since count() may be expensive.
            # Note that we call the Scala RDD API because the PySpark API does not respect the
            # specified timeout. Reference code:
            # https://spark.apache.org/docs/3.4.0/api/python/_modules/pyspark/rdd.html
            # #RDD.countApprox. This is confirmed to work in all Spark 3.x versions
            py_rdd = self.df.rdd
            drdd = py_rdd.mapPartitions(lambda it: [float(sum(1 for i in it))])
            jrdd = drdd.mapPartitions(lambda it: [float(sum(it))])._to_java_object_rdd()
            jdrdd = drdd.ctx._jvm.JavaDoubleRDD.fromRDD(jrdd.rdd())
            timeout_millis = 5000
            confidence = 0.9
            approx_count_operation = jdrdd.sumApprox(timeout_millis, confidence)
            approx_count_result = approx_count_operation.initialValue()
            approx_count_float = BoundedFloat(
                mean=approx_count_result.mean(),
                confidence=approx_count_result.confidence(),
                low=approx_count_result.low(),
                high=approx_count_result.high(),
            )
            approx_count = int(approx_count_float)
            if approx_count <= 0:
                # An approximate count of zero likely indicates that the count timed
                # out before an estimate could be made. In this case, we use the value
                # "unknown" so that users don't think the dataset is empty
                approx_count = "unknown"

            return {
                "approx_count": approx_count,
            }
        except Exception as e:
            _logger.warning(
                "Encountered an unexpected exception while computing Spark dataset profile."
                " Exception: %s",
                e,
            )

    @cached_property
    def schema(self) -> Optional[Schema]:
        """
        The MLflow ColSpec schema of the Spark dataset.
        """
        try:
            return _infer_schema(self._df)
        except Exception as e:
            _logger.warning("Failed to infer schema for Spark dataset. Exception: %s", e)
            return None

    def to_pyfunc(self) -> PyFuncInputsOutputs:
        """
        Converts the Spark DataFrame to pandas and splits the resulting
        :py:class:`pandas.DataFrame` into: 1. a :py:class:`pandas.DataFrame` of features and
        2. a :py:class:`pandas.Series` of targets.

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

    def to_evaluation_dataset(self, path=None, feature_names=None) -> EvaluationDataset:
        """
        Converts the dataset to an EvaluationDataset for model evaluation. Required
        for use with mlflow.evaluate().
        """
        return EvaluationDataset(
            data=self._df.limit(10000).toPandas(),
            targets=self._targets,
            path=path,
            feature_names=feature_names,
        )


@experimental
def load_delta(
    path: Optional[str] = None,
    table_name: Optional[str] = None,
    version: Optional[str] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> SparkDataset:
    """
    Loads a :py:class:`SparkDataset <mlflow.data.spark_dataset.SparkDataset>` from a Delta table
    for use with MLflow Tracking.

    :param path: The path to the Delta table. Either ``path`` or ``table_name`` must be specified.
    :param table_name: The name of the Delta table. Either ``path`` or ``table_name`` must be
                       specified.
    :param version: The Delta table version. If not specified, the version will be inferred.
    :param targets: Optional. The name of the Delta table column containing targets (labels) for
                    supervised learning.
    :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                 automatically generated.
    :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                   is automatically computed.
    :return: An instance of :py:class:`SparkDataset <mlflow.data.spark_dataset.SparkDataset>`.
    """
    from mlflow.data.spark_delta_utils import (
        _try_get_delta_table_latest_version_from_path,
        _try_get_delta_table_latest_version_from_table_name,
    )

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
        name = table_name + (f"@v{version}" if version is not None else "")

    source = DeltaDatasetSource(path=path, delta_table_name=table_name, delta_table_version=version)
    df = source.load()

    return SparkDataset(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
    )


@experimental
def from_spark(
    df: "pyspark.sql.DataFrame",
    path: Optional[str] = None,
    table_name: Optional[str] = None,
    version: Optional[str] = None,
    sql: Optional[str] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> SparkDataset:
    """
    Given a Spark DataFrame, constructs a
    :py:class:`SparkDataset <mlflow.data.spark_dataset.SparkDataset>` object for use with
    MLflow Tracking.

    :param df: The Spark DataFrame from which to construct a SparkDataset.
    :param path: The path of the Spark or Delta source that the DataFrame originally came from.
                 Note that the path does not have to match the DataFrame exactly, since the
                 DataFrame may have been modified by Spark operations. This is used to reload the
                 dataset upon request via :py:func:`SparkDataset.source.load()
                 <mlflow.data.spark_dataset_source.SparkDatasetSource.load>`. If none of ``path``,
                 ``table_name``, or ``sql`` are specified, a CodeDatasetSource is used, which will
                 source information from the run context.
    :param table_name: The name of the Spark or Delta table that the DataFrame originally came from.
                       Note that the table does not have to match the DataFrame exactly, since the
                       DataFrame may have been modified by Spark operations. This is used to reload
                       the dataset upon request via :py:func:`SparkDataset.source.load()
                       <mlflow.data.spark_dataset_source.SparkDatasetSource.load>`. If none of
                       ``path``, ``table_name``, or ``sql`` are specified, a CodeDatasetSource is
                       used, which will source information from the run context.
    :param version: If the DataFrame originally came from a Delta table, specifies the version
                    of the Delta table. This is used to reload the dataset upon request via
                    :py:func:`SparkDataset.source.load()
                    <mlflow.data.spark_dataset_source.SparkDatasetSource.load>`.  ``version``
                    cannot be specified if ``sql`` is specified.
    :param sql: The Spark SQL statement that was originally used to construct the DataFrame.
                Note that the Spark SQL statement does not have to match the DataFrame exactly,
                since the DataFrame may have been modified by Spark operations. This is used to
                reload the dataset upon request via :py:func:`SparkDataset.source.load()
                <mlflow.data.spark_dataset_source.SparkDatasetSource.load>`. If none of
                ``path``, ``table_name``, or ``sql`` are specified, a CodeDatasetSource is used,
                which will source information from the run context.
    :param targets: Optional. The name of the Data Frame column containing targets (labels) for
                    supervised learning.
    :param name: The name of the dataset. E.g. "wiki_train". If unspecified, a name is
                 automatically generated.
    :param digest: The digest (hash, fingerprint) of the dataset. If unspecified, a digest
                   is automatically computed.
    :return: An instance of :py:class:`SparkDataset <mlflow.data.spark_dataset.SparkDataset>`.
    """
    from mlflow.data.code_dataset_source import CodeDatasetSource
    from mlflow.data.spark_delta_utils import (
        _is_delta_table,
        _is_delta_table_path,
        _try_get_delta_table_latest_version_from_path,
        _try_get_delta_table_latest_version_from_table_name,
    )
    from mlflow.tracking.context import registry

    if (path, table_name, sql).count(None) < 2:
        raise MlflowException(
            "Must specify at most one of `path`, `table_name`, or `sql`.",
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
    else:
        context_tags = registry.resolve_tags()
        source = CodeDatasetSource(tags=context_tags)

    return SparkDataset(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
    )
