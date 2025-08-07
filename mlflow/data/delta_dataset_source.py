import logging
from typing import Any

from mlflow.data.dataset_source import DatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_managed_catalog_messages_pb2 import (
    GetTable,
    GetTableResponse,
)
from mlflow.protos.databricks_managed_catalog_service_pb2 import DatabricksUnityCatalogService
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import get_full_name_from_sc
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    call_endpoint,
    extract_api_info_for_service,
)
from mlflow.utils.string_utils import _backtick_quote

DATABRICKS_HIVE_METASTORE_NAME = "hive_metastore"
# these two catalog names both points to the workspace local default HMS (hive metastore).
DATABRICKS_LOCAL_METASTORE_NAMES = [DATABRICKS_HIVE_METASTORE_NAME, "spark_catalog"]
# samples catalog is managed by databricks for hosting public dataset like NYC taxi dataset.
# it is neither a UC nor local metastore catalog
DATABRICKS_SAMPLES_CATALOG_NAME = "samples"

_logger = logging.getLogger(__name__)


class DeltaDatasetSource(DatasetSource):
    """
    Represents the source of a dataset stored at in a delta table.
    """

    def __init__(
        self,
        path: str | None = None,
        delta_table_name: str | None = None,
        delta_table_version: int | None = None,
        delta_table_id: str | None = None,
    ):
        if (path, delta_table_name).count(None) != 1:
            raise MlflowException(
                'Must specify exactly one of "path" or "table_name"',
                INVALID_PARAMETER_VALUE,
            )
        self._path = path
        if delta_table_name is not None:
            self._delta_table_name = get_full_name_from_sc(
                delta_table_name, _get_active_spark_session()
            )
        else:
            self._delta_table_name = delta_table_name
        self._delta_table_version = delta_table_version
        self._delta_table_id = delta_table_id

    @staticmethod
    def _get_source_type() -> str:
        return "delta_table"

    def load(self, **kwargs):
        """
        Loads the dataset source as a Delta Dataset Source.

        Returns:
            An instance of ``pyspark.sql.DataFrame``.
        """
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()

        spark_read_op = spark.read.format("delta")
        if self._delta_table_version is not None:
            spark_read_op = spark_read_op.option("versionAsOf", self._delta_table_version)

        if self._path:
            return spark_read_op.load(self._path)
        else:
            backticked_delta_table_name = ".".join(
                map(_backtick_quote, self._delta_table_name.split("."))
            )
            return spark_read_op.table(backticked_delta_table_name)

    @property
    def path(self) -> str | None:
        return self._path

    @property
    def delta_table_name(self) -> str | None:
        return self._delta_table_name

    @property
    def delta_table_id(self) -> str | None:
        return self._delta_table_id

    @property
    def delta_table_version(self) -> int | None:
        return self._delta_table_version

    @staticmethod
    def _can_resolve(raw_source: Any):
        return False

    @classmethod
    def _resolve(cls, raw_source: str) -> "DeltaDatasetSource":
        raise NotImplementedError

    # check if table is in the Databricks Unity Catalog
    def _is_databricks_uc_table(self):
        if self._delta_table_name is not None:
            catalog_name = self._delta_table_name.split(".", 1)[0]
            return (
                catalog_name not in DATABRICKS_LOCAL_METASTORE_NAMES
                and catalog_name != DATABRICKS_SAMPLES_CATALOG_NAME
            )
        else:
            return False

    def _lookup_table_id(self, table_name):
        try:
            req_body = message_to_json(GetTable(full_name_arg=table_name))
            _METHOD_TO_INFO = extract_api_info_for_service(
                DatabricksUnityCatalogService, _REST_API_PATH_PREFIX
            )
            db_creds = get_databricks_host_creds()
            endpoint, method = _METHOD_TO_INFO[GetTable]
            # We need to replace the full_name_arg in the endpoint definition with
            # the actual table name for the REST API to work.
            final_endpoint = endpoint.replace("{full_name_arg}", table_name)
            resp = call_endpoint(
                host_creds=db_creds,
                endpoint=final_endpoint,
                method=method,
                json_body=req_body,
                response_proto=GetTableResponse,
            )
            return resp.table_id
        except Exception:
            return None

    def to_dict(self) -> dict[Any, Any]:
        info = {}
        if self._path:
            info["path"] = self._path
        if self._delta_table_name:
            info["delta_table_name"] = self._delta_table_name
        if self._delta_table_version:
            info["delta_table_version"] = self._delta_table_version
        if self._is_databricks_uc_table():
            info["is_databricks_uc_table"] = True
            if self._delta_table_id:
                info["delta_table_id"] = self._delta_table_id
            else:
                info["delta_table_id"] = self._lookup_table_id(self._delta_table_name)
        return info

    @classmethod
    def from_dict(cls, source_dict: dict[Any, Any]) -> "DeltaDatasetSource":
        return cls(
            path=source_dict.get("path"),
            delta_table_name=source_dict.get("delta_table_name"),
            delta_table_version=source_dict.get("delta_table_version"),
            delta_table_id=source_dict.get("delta_table_id"),
        )
