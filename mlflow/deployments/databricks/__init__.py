from typing import Any, Dict, Optional

from mlflow.deployments import BaseDeploymentClient
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request


class DatabricksDeploymentClient(BaseDeploymentClient):
    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        raise NotImplementedError

    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        raise NotImplementedError

    def delete_deployment(self, name, config=None, endpoint=None):
        raise NotImplementedError

    def list_deployments(self, endpoint=None):
        raise NotImplementedError

    def get_deployment(self, name, endpoint=None):
        raise NotImplementedError

    @property
    def _host_creds(self):
        """
        NB: When `MlflowGatewayClient` is used as an instance variable in a custom pyfunc model, it
        is pickled in the environment where the custom pyfunc model is defined (e.g. a notebook).
        When the model is moved to a different environment, e.g. model serving, new credentials
        need to be resolved from within the new environment. Accordingly, we re-resolve host
        credentials every time a request is made.
        """
        return get_databricks_host_creds("databricks")

    def _call_endpoint(
        self,
        method: str,
        prefix: str = "/api/2.0",
        route: Optional[str] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ):
        call_kwargs = {}
        if method.lower() == "get":
            call_kwargs["params"] = json_body
        else:
            call_kwargs["json"] = json_body

        response = http_request(
            host_creds=self._host_creds,
            endpoint=f"{prefix}/serving-endpoints" + (f"/{route}" if route else ""),
            method=method,
            timeout=10,
            retry_codes=frozenset(
                [
                    429,  # Too many requests
                    500,  # Server Error
                    502,  # Bad Gateway
                    503,  # Service Unavailable
                ]
            ),
            raise_on_status=False,
            **call_kwargs,
        )
        augmented_raise_for_status(response)
        return response.json()

    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        return self._call_endpoint(
            "POST", prefix="", route=f"{endpoint}/invocations", json_body=inputs
        )

    def create_endpoint(self, name, config=None):
        config = config.copy() if config else {}
        payload = {"name": name, "config": config}
        if task := config.pop("task", None):
            payload["task"] = task
        return self._call_endpoint("POST", json_body=payload)

    def update_endpoint(self, endpoint, config=None):
        return self._call_endpoint("PUT", route=f"{endpoint}/config", json_body=config)

    def delete_endpoint(self, endpoint):
        return self._call_endpoint("DELETE", route=endpoint)

    def list_endpoints(self):
        return self._call_endpoint("GET")

    def get_endpoint(self, endpoint):
        return self._call_endpoint("GET", route=endpoint)


def run_local(name, model_uri, flavor=None, config=None):
    pass


def target_help():
    pass
