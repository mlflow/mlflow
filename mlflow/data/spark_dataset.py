import json
import hashlib
from typing import Any, Dict, Optional

from pyspark.sql import SparkSession, DataFrame

from mlflow.data.dataset import Dataset
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.delta_dataset_source import DeltaDatasetSource
from mlflow.data.spark_dataset_source import SparkDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.types import Schema
from mlflow.types.utils import _infer_schema


class SparkDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        source: DatasetSource,
        targets: Optional[str] = None,
        name: Optional[str] = None,
        digest: Optional[str] = None,
    ):
        self._df = df
        super().__init__(source=source, name=name, digest=digest)

    @staticmethod
    def _parse_logical_plan(df):
        d = json.loads(df._jdf.queryExecution().logical().toJSON())

        def purge_key(input_dict, key):
            if isinstance(input_dict, dict):
                return {k: purge_key(v, key) for k, v in input_dict.items() if k != key}

            elif isinstance(input_dict, list):
                return [purge_key(element, key) for element in input_dict]

            else:
                return input_dict

        d = purge_key(d, "exprId")
        d = purge_key(d, "resultId")

        return d

    def _compute_digest(self) -> str:
        """
        Computes a digest for the dataset. Called if the user doesn't supply
        a digest when constructing the dataset.
        """
        parsed_plan = SparkDataset.parse_logical_plan(self._df)
        plan_str = json.dumps(parsed_plan)
        return hashlib.md5(plan_str.encode("utf-8")).hexdigest()[:8]

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
        return self._df

    @property
    def targets(self) -> Optional[str]:
        return self._targets

    @property
    def source(self) -> DatasetSource:
        return self._source

    @property
    def profile(self) -> Optional[Any]:
        # TODO: Include an approximate count of records from the table source. We don't
        # want to compute the full count, which could be quite slow
        return {}

    @property
    def schema(self) -> Schema:
        return _infer_schema(self._df)


def load_delta(
    path: Optional[str] = None,
    table_name: Optional[str] = None,
    version: Optional[str] = None,
    targets: Optional[str] = None,
    name: Optional[str] = None,
    digest: Optional[str] = None,
) -> SparkDataset:
    source = DeltaDatasetSource(path=path, table_name=table_name, version=version)
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
):
    if (path, table_name, sql).count(None) != 2:
        raise MlflowException(
            "Must specify exactly one of `path`, `table_name`, or `sql`.",
            INVALID_PARAMETER_VALUE,
        )

    if (sql, version).count(None) == 0:
        raise MlflowException(
            "`version` may only be specified when `table_name` or `path` is specified.",
            INVALID_PARAMETER_VALUE,
        )

    if sql is not None:
        source = SparkDatasetSource(sql=sql)
    elif path is not None:
        if _is_delta_table_path(path):
            source = DeltaDatasetSource(path=path, version=version)
        elif version is None:
            source = SparkDatasetSource(path=path)
        else:
            raise MlflowException(
                f"Version {version} was specified, but the path {path} does not refer"
                f" to a Delta table.",
                INVALID_PARAMETER_VALUE,
            )
    elif table_name is not None:
        if version is not None or _is_delta_table(table_name):
            source = DeltaDatasetSource(
                table_name=table_name,
                version=version,
            )
        else:
            source = SparkDatasetSource(table_name=table_name)

    return SparkDataset(
        df=df,
        source=source,
        targets=targets,
        name=name,
        digest=digest,
    )


def _is_delta_table(table_name: str) -> bool:
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
    if os.path.exists(path) and "_delta_log" in os.listdir(path):
        return True

    try:
        dbfs_path = dbfs_hdfs_uri_to_fuse_path(path)
        return os.path.exists(dbfs_path) and "_delta_log" in os.listdir(dbfs_path)
    except Exception:
        return False
