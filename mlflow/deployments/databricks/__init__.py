import posixpath
from typing import Any, Dict, Optional

from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments.constants import (
    MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
    MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT,
)
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.utils import AttrDict
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request


class DatabricksDeploymentClient(BaseDeploymentClient):
    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        .. warning::

            This method is not implemented for Databricks deployments.
        """
        raise NotImplementedError

    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        """
        .. warning::

            This method is not implemented for Databricks deployments.
        """
        raise NotImplementedError

    def delete_deployment(self, name, config=None, endpoint=None):
        """
        .. warning::

            This method is not implemented for Databricks deployments.
        """
        raise NotImplementedError

    def list_deployments(self, endpoint=None):
        """
        .. warning::

            This method is not implemented for Databricks deployments.
        """
        raise NotImplementedError

    def get_deployment(self, name, endpoint=None):
        """
        .. warning::

            This method is not implemented for Databricks deployments.
        """
        raise NotImplementedError

    def _call_endpoint(
        self,
        *,
        method: str,
        prefix: str = "/api/2.0",
        route: Optional[str] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: int = MLFLOW_HTTP_REQUEST_TIMEOUT.get(),
    ):
        call_kwargs = {}
        if method.lower() == "get":
            call_kwargs["params"] = json_body
        else:
            call_kwargs["json"] = json_body

        response = http_request(
            host_creds=get_databricks_host_creds(self.target_uri),
            endpoint=posixpath.join(prefix, "serving-endpoints", route or ""),
            method=method,
            timeout=timeout,
            raise_on_status=False,
            retry_codes=MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
            **call_kwargs,
        )
        augmented_raise_for_status(response)
        return AttrDict(response.json())

    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        """
        TODO
        """
        return self._call_endpoint(
            method="POST",
            prefix="/",
            route=posixpath.join(endpoint, "invocations"),
            json_body=inputs,
            timeout=MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT.get(),
        )

    def create_endpoint(self, name, config=None):
        """
        TODO
        """
        config = config.copy() if config else {}  # avoid mutating config
        payload = {"name": name, "config": config}
        if tags := config.pop("tags", None):
            payload["tags"] = tags
        return self._call_endpoint(method="POST", json_body=payload)

    def update_endpoint(self, endpoint, config=None):
        """
        TODO
        """
        route = "rate-limits" if "rate_limits" in config else "config"
        return self._call_endpoint(
            method="PUT", route=posixpath.join(endpoint, route), json_body=config
        )

    def delete_endpoint(self, endpoint):
        """
        TODO
        """
        return self._call_endpoint(method="DELETE", route=endpoint)

    def list_endpoints(self):
        """
        TODO
        """
        return self._call_endpoint(method="GET")

    def get_endpoint(self, endpoint):
        """
        TODO
        """
        return self._call_endpoint(method="GET", route=endpoint)


def run_local(name, model_uri, flavor=None, config=None):
    pass


def target_help():
    pass
