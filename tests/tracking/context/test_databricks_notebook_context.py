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
    patch_workspace_info = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils",
        return_value=("https://databricks.com", "123456"),
    )

    with multi_context(
        patch_notebook_id, patch_notebook_path, patch_webapp_url, patch_workspace_info
    ) as (
        notebook_id_mock,
        notebook_path_mock,
        webapp_url_mock,
        workspace_info_mock,
    ):
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: notebook_path_mock.return_value,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            MLFLOW_DATABRICKS_NOTEBOOK_ID: notebook_id_mock.return_value,
            MLFLOW_DATABRICKS_NOTEBOOK_PATH: notebook_path_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_URL: workspace_info_mock.return_value[0],
            MLFLOW_DATABRICKS_WORKSPACE_ID: workspace_info_mock.return_value[1],
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
