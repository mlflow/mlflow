from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import databricks_utils
from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_TYPE,
    MLFLOW_SOURCE_NAME,
    MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_JOB_RUN_ID,
    MLFLOW_DATABRICKS_JOB_TYPE,
    MLFLOW_DATABRICKS_WORKSPACE_URL,
    MLFLOW_DATABRICKS_WORKSPACE_ID,
    MLFLOW_DATABRICKS_GIT_URL,
    MLFLOW_DATABRICKS_GIT_PROVIDER,
    MLFLOW_DATABRICKS_GIT_COMMIT,
    MLFLOW_DATABRICKS_GIT_RELATIVE_PATH,
    MLFLOW_DATABRICKS_GIT_REFERENCE,
    MLFLOW_DATABRICKS_GIT_REFERENCE_TYPE,
    MLFLOW_DATABRICKS_GIT_STATUS,
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
        workspace_url_fallback, workspace_id = databricks_utils.get_workspace_info_from_dbutils()
        tags = {
            MLFLOW_SOURCE_NAME: (
                "jobs/{job_id}/run/{job_run_id}".format(job_id=job_id, job_run_id=job_run_id)
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
        elif workspace_url_fallback is not None:
            tags[MLFLOW_DATABRICKS_WORKSPACE_URL] = workspace_url_fallback
        if workspace_id is not None:
            tags[MLFLOW_DATABRICKS_WORKSPACE_ID] = workspace_id

        git_repo_url = databricks_utils.get_git_repo_url()
        git_repo_provider = databricks_utils.get_git_repo_provider()
        git_repo_commit = databricks_utils.get_git_repo_commit()
        git_repo_relative_path = databricks_utils.get_git_repo_relative_path()
        git_repo_reference = databricks_utils.get_git_repo_reference()
        git_repo_reference_type = databricks_utils.get_git_repo_reference_type()
        git_repo_status = databricks_utils.get_git_repo_status()

        if git_repo_url is not None:
            tags[MLFLOW_DATABRICKS_GIT_URL] = git_repo_url
        if git_repo_provider is not None:
            tags[MLFLOW_DATABRICKS_GIT_PROVIDER] = git_repo_provider
        if git_repo_commit is not None:
            tags[MLFLOW_DATABRICKS_GIT_COMMIT] = git_repo_commit
        if git_repo_relative_path is not None:
            tags[MLFLOW_DATABRICKS_GIT_RELATIVE_PATH] = git_repo_relative_path
        if git_repo_reference is not None:
            tags[MLFLOW_DATABRICKS_GIT_REFERENCE] = git_repo_reference
        if git_repo_reference_type is not None:
            tags[MLFLOW_DATABRICKS_GIT_REFERENCE_TYPE] = git_repo_reference_type
        if git_repo_status is not None:
            tags[MLFLOW_DATABRICKS_GIT_STATUS] = git_repo_status

        return tags
