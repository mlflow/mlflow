from unittest import mock

from mlflow.utils.mlflow_tags import MLFLOW_DATABRICKS_CLUSTER_ID
from mlflow.tracking.context.databricks_cluster_context import DatabricksClusterRunContext


def test_databricks_cluster_run_context_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_cluster") as in_cluster_mock:
        assert DatabricksClusterRunContext().in_context() == in_cluster_mock.return_value


def test_databricks_cluster_run_context_tags():
    patch_cluster_id = mock.patch("mlflow.utils.databricks_utils.get_cluster_id")
    with patch_cluster_id as cluster_id_mock:
        assert DatabricksClusterRunContext().tags() == {
            MLFLOW_DATABRICKS_CLUSTER_ID: cluster_id_mock.return_value
        }


def test_databricks_notebook_run_context_tags_nones():

    assert DatabricksClusterRunContext().tags() == {}
