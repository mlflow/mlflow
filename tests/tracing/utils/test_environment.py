from unittest import mock

import pytest

from mlflow.tracing.utils.environment import resolve_env_metadata
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_PATH,
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
    MLFLOW_SOURCE_NAME,
    MLFLOW_SOURCE_TYPE,
    MLFLOW_USER,
)
from mlflow.version import IS_TRACING_SDK_ONLY


@pytest.fixture(autouse=True)
def clear_lru_cache():
    resolve_env_metadata.cache_clear()


def test_resolve_env_metadata():
    expected_metadata = {
        MLFLOW_USER: mock.ANY,
        MLFLOW_SOURCE_NAME: mock.ANY,
        MLFLOW_SOURCE_TYPE: "LOCAL",
    }
    if not IS_TRACING_SDK_ONLY:
        expected_metadata.update(
            {
                MLFLOW_GIT_BRANCH: mock.ANY,
                MLFLOW_GIT_COMMIT: mock.ANY,
                MLFLOW_GIT_REPO_URL: mock.ANY,
            }
        )
    assert resolve_env_metadata() == expected_metadata


def test_resolve_env_metadata_in_databricks_notebook():
    with (
        mock.patch(
            "mlflow.tracking.context.databricks_notebook_context.databricks_utils"
        ) as mock_db_utils,
        mock.patch("mlflow.tracing.utils.environment.is_in_databricks_notebook", return_value=True),
    ):
        mock_db_utils.is_in_databricks_notebook.return_value = True
        mock_db_utils.get_notebook_id.return_value = "notebook_123"
        mock_db_utils.get_notebook_path.return_value = "/Users/bob/test.py"
        mock_db_utils.get_webapp_url.return_value = None
        mock_db_utils.get_workspace_url.return_value = None
        mock_db_utils.get_workspace_id.return_value = None
        mock_db_utils.get_workspace_info_from_dbutils.return_value = (None, None)

        assert resolve_env_metadata() == {
            MLFLOW_USER: mock.ANY,
            MLFLOW_SOURCE_NAME: "/Users/bob/test.py",
            MLFLOW_SOURCE_TYPE: "NOTEBOOK",
            MLFLOW_DATABRICKS_NOTEBOOK_ID: "notebook_123",
            MLFLOW_DATABRICKS_NOTEBOOK_PATH: "/Users/bob/test.py",
        }
