from mlflow.tracking.request_header.abstract_request_header_provider import RequestHeaderProvider
from mlflow.utils import databricks_utils


class DatabricksRequestHeaderProvider(RequestHeaderProvider):
    """
    Provides request headers indicating the type of Databricks environment from which a request
    was made.
    """

    def in_context(self):
        return (
            databricks_utils.is_in_cluster()
            or databricks_utils.is_in_databricks_notebook()
            or databricks_utils.is_in_databricks_job()
        )

    def request_headers(self):
        request_headers = {}
        if databricks_utils.is_in_databricks_notebook():
            request_headers["notebook_id"] = databricks_utils.get_notebook_id()
        if databricks_utils.is_in_databricks_job():
            request_headers["job_id"] = databricks_utils.get_job_id()
            request_headers["job_run_id"] = databricks_utils.get_job_run_id()
            request_headers["job_type"] = databricks_utils.get_job_type()
        if databricks_utils.is_in_cluster():
            request_headers["cluster_id"] = databricks_utils.get_cluster_id()
        command_run_id = databricks_utils.get_command_run_id()
        if command_run_id is not None:
            request_headers["command_run_id"] = command_run_id

        return request_headers
