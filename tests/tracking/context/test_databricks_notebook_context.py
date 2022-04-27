from unittest import mock

from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_PATH,
    MLFLOW_DATABRICKS_WEBAPP_URL,
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
from mlflow.tracking.context.databricks_notebook_context import DatabricksNotebookRunContext
from tests.helper_functions import multi_context


def test_databricks_notebook_run_context_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_notebook") as in_notebook_mock:
        assert DatabricksNotebookRunContext().in_context() == in_notebook_mock.return_value


def test_databricks_notebook_run_context_tags():
    patch_notebook_id = mock.patch("mlflow.utils.databricks_utils.get_notebook_id")
    patch_notebook_path = mock.patch("mlflow.utils.databricks_utils.get_notebook_path")
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url")
    patch_workspace_url = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_url",
        return_value="https://dev.databricks.com",
    )
    patch_workspace_url_none = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_url", return_value=None
    )
    patch_workspace_info = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils",
        return_value=("https://databricks.com", "123456"),
    )
    patch_git_repo_url = mock.patch("mlflow.utils.databricks_utils.get_git_repo_url")
    patch_git_repo_provider = mock.patch("mlflow.utils.databricks_utils.get_git_repo_provider")
    patch_git_repo_commit = mock.patch("mlflow.utils.databricks_utils.get_git_repo_commit")
    patch_git_repo_relative_path = mock.patch(
        "mlflow.utils.databricks_utils.get_git_repo_relative_path"
    )
    patch_git_repo_reference = mock.patch("mlflow.utils.databricks_utils.get_git_repo_reference")
    patch_git_repo_reference_type = mock.patch(
        "mlflow.utils.databricks_utils.get_git_repo_reference_type"
    )
    patch_git_repo_status = mock.patch("mlflow.utils.databricks_utils.get_git_repo_status")

    with multi_context(
        patch_notebook_id,
        patch_notebook_path,
        patch_webapp_url,
        patch_workspace_url,
        patch_workspace_info,
        patch_git_repo_url,
        patch_git_repo_provider,
        patch_git_repo_commit,
        patch_git_repo_relative_path,
        patch_git_repo_reference,
        patch_git_repo_reference_type,
        patch_git_repo_status,
    ) as (
        notebook_id_mock,
        notebook_path_mock,
        webapp_url_mock,
        workspace_url_mock,
        workspace_info_mock,
        git_repo_url_mock,
        git_repo_provider_mock,
        git_repo_commit_mock,
        git_repo_relative_path_mock,
        git_repo_reference_mock,
        git_repo_reference_type_mock,
        git_repo_status_mock,
    ):
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: notebook_path_mock.return_value,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            MLFLOW_DATABRICKS_NOTEBOOK_ID: notebook_id_mock.return_value,
            MLFLOW_DATABRICKS_NOTEBOOK_PATH: notebook_path_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_URL: workspace_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_ID: workspace_info_mock.return_value[1],
            MLFLOW_DATABRICKS_GIT_URL: git_repo_url_mock.return_value,
            MLFLOW_DATABRICKS_GIT_PROVIDER: git_repo_provider_mock.return_value,
            MLFLOW_DATABRICKS_GIT_COMMIT: git_repo_commit_mock.return_value,
            MLFLOW_DATABRICKS_GIT_RELATIVE_PATH: git_repo_relative_path_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REFERENCE: git_repo_reference_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REFERENCE_TYPE: git_repo_reference_type_mock.return_value,
            MLFLOW_DATABRICKS_GIT_STATUS: git_repo_status_mock.return_value,
        }

    with multi_context(
        patch_notebook_id,
        patch_notebook_path,
        patch_webapp_url,
        patch_workspace_url_none,
        patch_workspace_info,
        patch_git_repo_url,
        patch_git_repo_provider,
        patch_git_repo_commit,
        patch_git_repo_relative_path,
        patch_git_repo_reference,
        patch_git_repo_reference_type,
        patch_git_repo_status,
    ) as (
        notebook_id_mock,
        notebook_path_mock,
        webapp_url_mock,
        workspace_url_mock,
        workspace_info_mock,
        git_repo_url_mock,
        git_repo_provider_mock,
        git_repo_commit_mock,
        git_repo_relative_path_mock,
        git_repo_reference_mock,
        git_repo_reference_type_mock,
        git_repo_status_mock,
    ):
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: notebook_path_mock.return_value,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            MLFLOW_DATABRICKS_NOTEBOOK_ID: notebook_id_mock.return_value,
            MLFLOW_DATABRICKS_NOTEBOOK_PATH: notebook_path_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_URL: workspace_info_mock.return_value[0],  # fallback value
            MLFLOW_DATABRICKS_WORKSPACE_ID: workspace_info_mock.return_value[1],
            MLFLOW_DATABRICKS_GIT_URL: git_repo_url_mock.return_value,
            MLFLOW_DATABRICKS_GIT_PROVIDER: git_repo_provider_mock.return_value,
            MLFLOW_DATABRICKS_GIT_COMMIT: git_repo_commit_mock.return_value,
            MLFLOW_DATABRICKS_GIT_RELATIVE_PATH: git_repo_relative_path_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REFERENCE: git_repo_reference_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REFERENCE_TYPE: git_repo_reference_type_mock.return_value,
            MLFLOW_DATABRICKS_GIT_STATUS: git_repo_status_mock.return_value,
        }


def test_databricks_notebook_run_context_tags_nones():
    patch_notebook_id = mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_id", return_value=None
    )
    patch_notebook_path = mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_path", return_value=None
    )
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url", return_value=None)
    patch_workspace_info = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils", return_value=(None, None)
    )

    with patch_notebook_id, patch_notebook_path, patch_webapp_url, patch_workspace_info:
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: None,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        }
