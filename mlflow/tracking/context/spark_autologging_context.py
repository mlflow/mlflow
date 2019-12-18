import threading

from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import databricks_utils
from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_TYPE,
    MLFLOW_SOURCE_NAME,
    MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_DATABRICKS_NOTEBOOK_PATH,
    MLFLOW_DATABRICKS_NOTEBOOK_ID
)

_SPARK_TABLE_INFO_TAG_NAME = "sparkTableInfo"

_lock = threading.Lock()
_table_infos = []

def _get_table_info_string(path, version, format):
    if format == "delta":
        return "path={path},version={version},format={format}".format(
            path=path, version=version, format=format)
    return "path={path},format={format}".format(path=path, format=format)

def _merge_tag_lines(existing_tag, new_table_info):
    if existing_tag is None:
        return new_table_info
    if new_table_info in existing_tag:
        return existing_tag
    return "\n".join([existing_tag, new_table_info])

def add_table_info(path, version, format):
    with _lock:
        _table_infos.append((path, version, format))

class SparkAutologgingContext(RunContextProvider):

    def in_context(self):
        return True

    def tags(self):
        with _lock:
            global _table_infos
            tags = {
                _SPARK_TABLE_INFO_TAG_NAME: "\n".join([_get_table_info_string(*info)
                                                       for info in _table_infos])
            }
            _table_infos = []
            return tags

