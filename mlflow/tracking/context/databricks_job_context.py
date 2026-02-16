from mlflow.entities import SourceType
from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import databricks_utils
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_JOB_RUN_ID,
    MLFLOW_DATABRICKS_JOB_TYPE,
    MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_DATABRICKS_WORKSPACE_ID,
    MLFLOW_DATABRICKS_WORKSPACE_URL,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
)


class DatabricksJobRunContext(RunContextProvider):
    def in_context(self):
        return databricks_utils.is_in_databricks_job()

    def tags(self):
        job_id = databricks_utils.get_job_id()
        job_run_id = databricks_utils.get_job_run_id()
        job_type = databricks_utils.get_job_type()
        webapp_url = databricks_utils.get_webapp_url()
        workspace_url = databricks_utils.get_workspace_url()
        workspace_id = databricks_utils.get_workspace_id()
        tags = {
            MLFLOW_SOURCE_NAME: (
                f"jobs/{job_id}/run/{job_run_id}"
                if job_id is not None and job_run_id is not None
                else None
            ),
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.JOB),
        }
        if job_id is not None:
            tags[MLFLOW_DATABRICKS_JOB_ID] = job_id
        if job_run_id is not None:
            tags[MLFLOW_DATABRICKS_JOB_RUN_ID] = job_run_id
        if job_type is not None:
            tags[MLFLOW_DATABRICKS_JOB_TYPE] = job_type
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
