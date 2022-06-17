from unittest import mock

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow import MlflowClient
from mlflow.tracking.default_experiment.databricks_notebook_experiment_provider import (
    DatabricksNotebookExperimentProvider,
    DatabricksRepoNotebookExperimentProvider,
)
from mlflow.utils.mlflow_tags import MLFLOW_EXPERIMENT_SOURCE_TYPE, MLFLOW_EXPERIMENT_SOURCE_ID


def test_databricks_notebook_default_experiment_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_notebook") as in_notebook_mock:
        assert DatabricksNotebookExperimentProvider().in_context() == in_notebook_mock.return_value


def test_databricks_notebook_default_experiment_id():
    with mock.patch("mlflow.utils.databricks_utils.get_notebook_id") as patch_notebook_id:
        assert (
            DatabricksNotebookExperimentProvider().get_experiment_id()
            == patch_notebook_id.return_value
        )


def test_databricks_repo_notebook_default_experiment_in_context():
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_repo_notebook"
    ) as in_repo_notebook_mock:
        in_repo_notebook_mock.return_value = True
        assert DatabricksRepoNotebookExperimentProvider().in_context()
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_repo_notebook"
    ) as not_in_repo_notebook_mock:
        not_in_repo_notebook_mock.return_value = False
        assert not DatabricksRepoNotebookExperimentProvider().in_context()


def test_databricks_repo_notebook_default_experiment_gets_id_by_request():
    with mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_id"
    ) as notebook_id_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_path"
    ) as notebook_path_mock, mock.patch.object(
        MlflowClient, "create_experiment"
    ) as create_experiment_mock:
        notebook_id_mock.return_value = 1234
        notebook_path_mock.return_value = "/Repos/path"
        create_experiment_mock.return_value = "experiment_id"
        returned_id = DatabricksRepoNotebookExperimentProvider().get_experiment_id()
        assert returned_id == "experiment_id"
        tags = {MLFLOW_EXPERIMENT_SOURCE_TYPE: "REPO_NOTEBOOK", MLFLOW_EXPERIMENT_SOURCE_ID: 1234}
        create_experiment_mock.assert_called_once_with("/Repos/path", None, tags)


def test_databricks_repo_notebook_default_experiment_uses_fallback_notebook_id():
    with mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_id"
    ) as notebook_id_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_path"
    ) as notebook_path_mock, mock.patch.object(
        MlflowClient, "create_experiment"
    ) as create_experiment_mock:
        DatabricksRepoNotebookExperimentProvider._resolved_repo_notebook_experiment_id = None
        notebook_id_mock.return_value = 1234
        notebook_path_mock.return_value = "/Repos/path"
        create_experiment_mock.side_effect = MlflowException(
            message="not enabled", error_code=INVALID_PARAMETER_VALUE
        )
        returned_id = DatabricksRepoNotebookExperimentProvider().get_experiment_id()
        assert returned_id == 1234
