from unittest import mock

from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_GIT_REPO_URL,
    MLFLOW_DATABRICKS_GIT_REPO_PROVIDER,
    MLFLOW_DATABRICKS_GIT_REPO_COMMIT,
    MLFLOW_DATABRICKS_GIT_REPO_RELATIVE_PATH,
    MLFLOW_DATABRICKS_GIT_REPO_REFERENCE,
    MLFLOW_DATABRICKS_GIT_REPO_REFERENCE_TYPE,
    MLFLOW_DATABRICKS_GIT_REPO_STATUS,
)
from mlflow.tracking.context.databricks_repo_context import DatabricksRepoRunContext
from tests.helper_functions import multi_context


def test_databricks_repo_run_context_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_repo") as in_repo_mock:
        assert DatabricksRepoRunContext().in_context() == in_repo_mock.return_value


def test_databricks_repo_run_context_tags():
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
        patch_git_repo_url,
        patch_git_repo_provider,
        patch_git_repo_commit,
        patch_git_repo_relative_path,
        patch_git_repo_reference,
        patch_git_repo_reference_type,
        patch_git_repo_status,
    ) as (
        git_repo_url_mock,
        git_repo_provider_mock,
        git_repo_commit_mock,
        git_repo_relative_path_mock,
        git_repo_reference_mock,
        git_repo_reference_type_mock,
        git_repo_status_mock,
    ):
        assert DatabricksRepoRunContext().tags() == {
            MLFLOW_DATABRICKS_GIT_REPO_URL: git_repo_url_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REPO_PROVIDER: git_repo_provider_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REPO_COMMIT: git_repo_commit_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REPO_RELATIVE_PATH: git_repo_relative_path_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REPO_REFERENCE: git_repo_reference_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REPO_REFERENCE_TYPE: git_repo_reference_type_mock.return_value,
            MLFLOW_DATABRICKS_GIT_REPO_STATUS: git_repo_status_mock.return_value,
        }


def test_databricks_repo_run_context_tags_nones():
    patch_git_repo_url = mock.patch(
        "mlflow.utils.databricks_utils.get_git_repo_url", return_value=None
    )
    patch_git_repo_provider = mock.patch(
        "mlflow.utils.databricks_utils.get_git_repo_provider", return_value=None
    )
    patch_git_repo_commit = mock.patch(
        "mlflow.utils.databricks_utils.get_git_repo_commit", return_value=None
    )
    patch_git_repo_relative_path = mock.patch(
        "mlflow.utils.databricks_utils.get_git_repo_relative_path", return_value=None
    )
    patch_git_repo_reference = mock.patch(
        "mlflow.utils.databricks_utils.get_git_repo_reference", return_value=None
    )
    patch_git_repo_reference_type = mock.patch(
        "mlflow.utils.databricks_utils.get_git_repo_reference_type", return_value=None
    )
    patch_git_repo_status = mock.patch(
        "mlflow.utils.databricks_utils.get_git_repo_status", return_value=None
    )
    with multi_context(
        patch_git_repo_url,
        patch_git_repo_provider,
        patch_git_repo_commit,
        patch_git_repo_relative_path,
        patch_git_repo_reference,
        patch_git_repo_reference_type,
        patch_git_repo_status,
    ):
        assert DatabricksRepoRunContext().tags() == {}
