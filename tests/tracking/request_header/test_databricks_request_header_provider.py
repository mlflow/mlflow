import pytest
import itertools
from unittest import mock

from mlflow.tracking.request_header.databricks_request_header_provider import (
    DatabricksRequestHeaderProvider,
)

bool_values = [True, False]


@pytest.mark.parametrize(
    "is_in_databricks_notebook,is_in_databricks_job,is_in_cluster",
    itertools.product(bool_values, bool_values, bool_values),
)
def test_databricks_request_header_provider_in_context(
    is_in_databricks_notebook, is_in_databricks_job, is_in_cluster
):
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_notebook",
        return_value=is_in_databricks_notebook,
    ), mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_job", return_value=is_in_databricks_job
    ), mock.patch(
        "mlflow.utils.databricks_utils.is_in_cluster", return_value=is_in_cluster
    ):
        assert (
            DatabricksRequestHeaderProvider().in_context() == is_in_databricks_notebook
            or is_in_databricks_job
            or is_in_cluster
        )


# test that request_headers returns whatever is available
@pytest.mark.parametrize(
    "is_in_databricks_notebook,is_in_databricks_job,is_in_cluster",
    itertools.product(bool_values, bool_values, bool_values),
)
def test_databricks_request_header_provider_request_headers(
    is_in_databricks_notebook, is_in_databricks_job, is_in_cluster
):
    with mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_notebook",
        return_value=is_in_databricks_notebook,
    ), mock.patch(
        "mlflow.utils.databricks_utils.is_in_databricks_job", return_value=is_in_databricks_job
    ), mock.patch(
        "mlflow.utils.databricks_utils.is_in_cluster", return_value=is_in_cluster
    ), mock.patch(
        "mlflow.utils.databricks_utils.get_notebook_id"
    ) as notebook_id_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_job_id"
    ) as job_id_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_job_run_id"
    ) as job_run_id_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_job_type"
    ) as job_type_mock, mock.patch(
        "mlflow.utils.databricks_utils.get_cluster_id"
    ) as cluster_id_mock:
        request_headers = DatabricksRequestHeaderProvider().request_headers()

        if is_in_databricks_notebook:
            assert request_headers["notebook_id"] == notebook_id_mock.return_value
        else:
            assert "notebook_id" not in request_headers

        if is_in_databricks_job:
            assert request_headers["job_id"] == job_id_mock.return_value
            assert request_headers["job_run_id"] == job_run_id_mock.return_value
            assert request_headers["job_type"] == job_type_mock.return_value
        else:
            assert "job_id" not in request_headers
            assert "job_run_id" not in request_headers
            assert "job_type" not in request_headers

        if is_in_cluster:
            assert request_headers["cluster_id"] == cluster_id_mock.return_value
        else:
            assert "cluster_id" not in request_headers
