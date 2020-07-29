import mock

from mlflow.entities import SourceType
from mlflow.utils.mlflow_tags import (
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_PATH,
    MLFLOW_DATABRICKS_WEBAPP_URL,
)
from mlflow.tracking.context.databricks_notebook_context import DatabricksNotebookRunContext


def test_databricks_notebook_run_context_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_notebook") as in_notebook_mock:
        assert DatabricksNotebookRunContext().in_context() == in_notebook_mock.return_value


def test_databricks_notebook_run_context_tags():
    patch_notebook_id = mock.patch("mlflow.utils.databricks_utils.get_notebook_id")
    patch_notebook_path = mock.patch("mlflow.utils.databricks_utils.get_notebook_path")
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url")

    with patch_notebook_id as notebook_id_mock, patch_notebook_path as notebook_path_mock, patch_webapp_url as webapp_url_mock:
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: notebook_path_mock.return_value,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
            MLFLOW_DATABRICKS_NOTEBOOK_ID: notebook_id_mock.return_value,
            MLFLOW_DATABRICKS_NOTEBOOK_PATH: notebook_path_mock.return_value,
            MLFLOW_DATABRICKS_WEBAPP_URL: webapp_url_mock.return_value,
        }


def test_databricks_notebook_run_context_tags_nones():
    patch_notebook_id = mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_id", return_value=None
    )
    patch_notebook_path = mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_path", return_value=None
    )
    patch_webapp_url = mock.patch("mlflow.utils.databricks_utils.get_webapp_url", return_value=None)

    with patch_notebook_id, patch_notebook_path, patch_webapp_url:
        assert DatabricksNotebookRunContext().tags() == {
            MLFLOW_SOURCE_NAME: None,
            MLFLOW_SOURCE_TYPE: SourceType.to_string(SourceType.NOTEBOOK),
        }
