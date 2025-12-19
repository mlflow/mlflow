from mlflow.entities import SourceType
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import databricks_utils
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_PATH,
    MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_DATABRICKS_WORKSPACE_ID,
    MLFLOW_DATABRICKS_WORKSPACE_URL,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
)


class DatabricksNotebookRunContext(RunContextProvider):
    def in_context(self):
        return databricks_utils.is_in_databricks_notebook()

    def tags(self):
        notebook_id = databricks_utils.get_notebook_id()
        notebook_path = databricks_utils.get_notebook_path()
        webapp_url = databricks_utils.get_webapp_url()
        workspace_url = databricks_utils.get_workspace_url()
        workspace_id = databricks_utils.get_workspace_id()
        tags = {
            MLFLOW_SOURCE_NAME: notebook_path,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        }
        if notebook_id is not None:
            tags[MLFLOW_DATABRICKS_NOTEBOOK_ID] = notebook_id
        if notebook_path is not None:
            tags[MLFLOW_DATABRICKS_NOTEBOOK_PATH] = notebook_path
        if webapp_url is not None:
            tags[MLFLOW_DATABRICKS_WEBAPP_URL] = webapp_url
        if workspace_url is not None:
            tags[MLFLOW_DATABRICKS_WORKSPACE_URL] = workspace_url
        else:
            workspace_url_fallback, _ = databricks_utils.get_workspace_info_from_dbutils()
            if workspace_url_fallback is not None:
                tags[MLFLOW_DATABRICKS_WORKSPACE_URL] = workspace_url_fallback
        if workspace_id is not None:
            tags[MLFLOW_DATABRICKS_WORKSPACE_ID] = workspace_id
        return tags
