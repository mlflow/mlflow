from unittest import mock

from mlflow.entities import SourceType
from mlflow.tracking.context.databricks_notebook_context import DatabricksNotebookRunContext
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_PATH,
    MLFLOW_DATABRICKS_WEBAPP_URL,
    MLFLOW_DATABRICKS_WORKSPACE_ID,
    MLFLOW_DATABRICKS_WORKSPACE_URL,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
)


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
    patch_workspace_id = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_id", return_value="123456"
    )
    patch_workspace_url_none = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_url", return_value=None
    )
    patch_workspace_info = mock.patch(
        "mlflow.utils.databricks_utils.get_workspace_info_from_dbutils",
        return_value=("https://databricks.com", "123456"),
    )

    with (
        patch_notebook_id as notebook_id_mock,
        patch_notebook_path as notebook_path_mock,
        patch_webapp_url as webapp_url_mock,
        patch_workspace_url as workspace_url_mock,
        patch_workspace_info as workspace_info_mock,
        patch_workspace_id as workspace_id_mock,
    ):
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: notebook_path_mock.return_value,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            MLFLOW_DATABRICKS_NOTEBOOK_ID: notebook_id_mock.return_value,
            MLFLOW_DATABRICKS_NOTEBOOK_PATH: notebook_path_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_URL: workspace_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_ID: workspace_id_mock.return_value,
        }

    with (
        patch_notebook_id as notebook_id_mock,
        patch_notebook_path as notebook_path_mock,
        patch_webapp_url as webapp_url_mock,
        patch_workspace_url_none as workspace_url_mock,
        patch_workspace_info as workspace_info_mock,
        patch_workspace_id as workspace_id_mock,
    ):
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: notebook_path_mock.return_value,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            MLFLOW_DATABRICKS_NOTEBOOK_ID: notebook_id_mock.return_value,
            MLFLOW_DATABRICKS_NOTEBOOK_PATH: notebook_path_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value,
            MLFLOW_DATABRICKS_WORKSPACE_URL: workspace_info_mock.return_value[0],  # fallback value
            MLFLOW_DATABRICKS_WORKSPACE_ID: workspace_id_mock.return_value,
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
