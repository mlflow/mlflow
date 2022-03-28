from unittest import mock

from mlflow.tracking.default_experiment.databricks_notebook_experiment_provider import (
    DatabricksNotebookExperimentProvider,
)


def test_databricks_notebook_default_experiment_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_databricks_notebook") as in_notebook_mock:
        assert DatabricksNotebookExperimentProvider().in_context() == in_notebook_mock.return_value


def test_databricks_notebook_default_experiment_id():
    with mock.patch("mlflow.utils.databricks_utils.get_notebook_id") as patch_notebook_id:
        assert (
            DatabricksNotebookExperimentProvider().get_experiment_id()
            == patch_notebook_id.return_value
        )
