from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils import databricks_utils
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_GIT_REPO_URL,
    MLFLOW_DATABRICKS_GIT_REPO_PROVIDER,
    MLFLOW_DATABRICKS_GIT_REPO_COMMIT,
    MLFLOW_DATABRICKS_GIT_REPO_RELATIVE_PATH,
    MLFLOW_DATABRICKS_GIT_REPO_REFERENCE,
    MLFLOW_DATABRICKS_GIT_REPO_REFERENCE_TYPE,
    MLFLOW_DATABRICKS_GIT_REPO_STATUS,
)


class DatabricksRepoRunContext(RunContextProvider):
    def in_context(self):
        return databricks_utils.is_in_databricks_repo()

    def tags(self):
        tags = {}
        git_repo_url = databricks_utils.get_git_repo_url()
        git_repo_provider = databricks_utils.get_git_repo_provider()
        git_repo_commit = databricks_utils.get_git_repo_commit()
        git_repo_relative_path = databricks_utils.get_git_repo_relative_path()
        git_repo_reference = databricks_utils.get_git_repo_reference()
        git_repo_reference_type = databricks_utils.get_git_repo_reference_type()
        git_repo_status = databricks_utils.get_git_repo_status()

        if git_repo_url is not None:
            tags[MLFLOW_DATABRICKS_GIT_REPO_URL] = git_repo_url
        if git_repo_provider is not None:
            tags[MLFLOW_DATABRICKS_GIT_REPO_PROVIDER] = git_repo_provider
        if git_repo_commit is not None:
            tags[MLFLOW_DATABRICKS_GIT_REPO_COMMIT] = git_repo_commit
        if git_repo_relative_path is not None:
            tags[MLFLOW_DATABRICKS_GIT_REPO_RELATIVE_PATH] = git_repo_relative_path
        if git_repo_reference is not None:
            tags[MLFLOW_DATABRICKS_GIT_REPO_REFERENCE] = git_repo_reference
        if git_repo_reference_type is not None:
            tags[MLFLOW_DATABRICKS_GIT_REPO_REFERENCE_TYPE] = git_repo_reference_type
        if git_repo_status is not None:
            tags[MLFLOW_DATABRICKS_GIT_REPO_STATUS] = git_repo_status

        return tags
