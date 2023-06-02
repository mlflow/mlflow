from typing import Any, Optional, Dict

from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.annotations import experimental


DATABRICKS_HIVE_METASTORE_NAME = "hive_metastore"
# these two catalog names both points to the workspace local default HMS (hive metastore).
DATABRICKS_LOCAL_METASTORE_NAMES = [DATABRICKS_HIVE_METASTORE_NAME, "spark_catalog"]
# samples catalog is managed by databricks for hosting public dataset like NYC taxi dataset.
# it is neither a UC nor local metastore catalog
DATABRICKS_SAMPLES_CATALOG_NAME = "samples"


@experimental
class DeltaDatasetSource(DatasetSource):
    """
    Represents the source of a dataset stored at in a delta table.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        delta_table_name: Optional[str] = None,
        delta_table_version: Optional[int] = None,
    ):
        if (path, delta_table_name).count(None) != 1:
            raise MlflowException(
                'Must specify exactly one of "path" or "table_name"',
                INVALID_PARAMETER_VALUE,
            )
        self._path = path
        self._delta_table_name = delta_table_name
        self._delta_table_version = delta_table_version

    @staticmethod
    def _get_source_type() -> str:
        return "delta_table"

    def load(self, **kwargs):
        """
        Loads the dataset source as a Delta Dataset Source.
        :return: An instance of ``pyspark.sql.DataFrame``.
        """
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        spark_read_op = spark.read.format("delta")
        if self._delta_table_version is not None:
            spark_read_op = spark_read_op.option("versionAsOf", self._delta_table_version)

        if self._path:
            return spark_read_op.load(self._path)
        else:
            return spark_read_op.table(self._delta_table_name)

    @property
    def path(self) -> Optional[str]:
        return self._path

    @property
    def delta_table_name(self) -> Optional[str]:
        return self._delta_table_name

    @property
    def delta_table_version(self) -> Optional[int]:
        return self._delta_table_version

    @staticmethod
    def _can_resolve(raw_source: Any):
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> "DeltaDatasetSource":
        raise NotImplementedError

    # check if table is in the Databricks Unity Catalog
    def _is_databricks_uc_table(self):
        if is_in_databricks_runtime() and self._delta_table_name is not None:
            catalog_name = self._delta_table_name.split(".", 1)[0]
            return (
                catalog_name not in DATABRICKS_LOCAL_METASTORE_NAMES
                and catalog_name != DATABRICKS_SAMPLES_CATALOG_NAME
            )

    def _to_dict(self) -> Dict[Any, Any]:
        info = {}
        if self._path:
            info["path"] = self._path
        if self._delta_table_name:
            info["delta_table_name"] = self._delta_table_name
        if self._delta_table_version:
            info["delta_table_version"] = self._delta_table_version
        if self._is_databricks_uc_table():
            info["is_databricks_uc_table"] = True
        return info

    @classmethod
    def _from_dict(cls, source_dict: Dict[Any, Any]) -> "DeltaDatasetSource":
        return cls(
            path=source_dict.get("path"),
            delta_table_name=source_dict.get("delta_table_name"),
            delta_table_version=source_dict.get("delta_table_version"),
        )
