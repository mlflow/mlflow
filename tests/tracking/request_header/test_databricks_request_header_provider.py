import pytest
from unittest import mock

from mlflow.tracking.request_header.databricks_request_header_provider import (
    DatabricksRequestHeaderProvider,
)


def test_databricks_cluster_run_context_in_context():
    with mock.patch("mlflow.utils.databricks_utils.is_in_cluster") as in_cluster_mock:
        assert DatabricksRequestHeaderProvider().in_context() == in_cluster_mock.return_value


@pytest.mark.parametrize("is_in_databricks_notebook", [True, False])
def test_notebook_request_headers(is_in_databricks_notebook):
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_notebook",
        return_value=is_in_databricks_notebook,
    ), mock.patch("mlflow.utils.databricks_utils.get_notebook_id") as notebook_id_mock:
        if is_in_databricks_notebook:
            assert (
                DatabricksRequestHeaderProvider().request_headers()["notebook_id"]
                == notebook_id_mock.return_value
            )
        else:
            assert "notebook_id" not in DatabricksRequestHeaderProvider().request_headers()


@pytest.mark.parametrize("is_in_databricks_job", [True, False])
def test_job_request_headers(is_in_databricks_job):
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_job", return_value=is_in_databricks_job
    ), mock.patch("mlflow.utils.databricks_utils.get_job_id") as job_id_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_job_run_id"
    ) as job_run_id_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_job_type"
    ) as job_type_mock:
        if is_in_databricks_job:
            assert (
                DatabricksRequestHeaderProvider().request_headers()["job_id"]
                == job_id_mock.return_value
            )
            assert (
                DatabricksRequestHeaderProvider().request_headers()["job_run_id"]
                == job_run_id_mock.return_value
            )
            assert (
                DatabricksRequestHeaderProvider().request_headers()["job_type"]
                == job_type_mock.return_value
            )
        else:
            assert "job_id" not in DatabricksRequestHeaderProvider().request_headers()
            assert "job_run_id" not in DatabricksRequestHeaderProvider().request_headers()
            assert "job_type" not in DatabricksRequestHeaderProvider().request_headers()


@pytest.mark.parametrize("is_in_cluster", [True, False])
def test_notebook_request_headers(is_in_cluster):
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_cluster", return_value=is_in_cluster,
    ), mock.patch("mlflow.utils.databricks_utils.get_cluster_id") as cluster_id_mock:
        if is_in_cluster:
            assert (
                DatabricksRequestHeaderProvider().request_headers()["cluster_id"]
                == cluster_id_mock.return_value
            )
        else:
            assert "cluster_id" not in DatabricksRequestHeaderProvider().request_headers()
