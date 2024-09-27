from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests

from mlflow import MlflowException
from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments.constants import (
    MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
)
from mlflow.deployments.server.constants import (
    MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE,
    MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE,
    MLFLOW_DEPLOYMENTS_QUERY_SUFFIX,
)
from mlflow.deployments.utils import resolve_endpoint_url
from mlflow.environment_variables import (
    MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
)
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.annotations import experimental
from mlflow.utils.credentials import get_default_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request
from mlflow.utils.uri import join_paths

if TYPE_CHECKING:
    from mlflow.deployments.server.config import Endpoint


@experimental
class MlflowDeploymentClient(BaseDeploymentClient):
    """
    Client for interacting with the MLflow AI Gateway.

    Example:

    First, start the MLflow AI Gateway:

    .. code-block:: bash

        mlflow gateway start --config-path path/to/config.yaml

    Then, create a client and use it to interact with the server:

    .. code-block:: python

        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("http://localhost:5000")
        endpoints = client.list_endpoints()
        assert [e.dict() for e in endpoints] == [
            {
                "name": "chat",
                "endpoint_type": "llm/v1/chat",
                "model": {"name": "gpt-4o-mini", "provider": "openai"},
                "endpoint_url": "http://localhost:5000/gateway/chat/invocations",
            },
        ]
    """

    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MlflowDeploymentClient`.
        """
        raise NotImplementedError

    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MlflowDeploymentClient`.
        """
        raise NotImplementedError

    def delete_deployment(self, name, config=None, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MlflowDeploymentClient`.
        """
        raise NotImplementedError

    def list_deployments(self, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MlflowDeploymentClient`.
        """
        raise NotImplementedError

    def get_deployment(self, name, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def create_endpoint(self, name, config=None):
        """
        .. warning::
            This method is not implemented for `MlflowDeploymentClient`.
        """
        raise NotImplementedError

    def update_endpoint(self, endpoint, config=None):
        """
        .. warning::
            This method is not implemented for `MlflowDeploymentClient`.
        """
        raise NotImplementedError

    def delete_endpoint(self, endpoint):
        """
        .. warning::
            This method is not implemented for `MlflowDeploymentClient`.
        """
        raise NotImplementedError

    def _call_endpoint(
        self,
        method: str,
        route: str,
        json_body: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        call_kwargs = {}
        if method.lower() == "get":
            call_kwargs["params"] = json_body
        else:
            call_kwargs["json"] = json_body

        response = http_request(
            host_creds=get_default_host_creds(self.target_uri),
            endpoint=route,
            method=method,
            timeout=MLFLOW_HTTP_REQUEST_TIMEOUT.get() if timeout is None else timeout,
            retry_codes=MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
            raise_on_status=False,
            **call_kwargs,
        )
        augmented_raise_for_status(response)
        return response.json()

    @experimental
    def get_endpoint(self, endpoint) -> "Endpoint":
        """
        Gets a specified endpoint configured for the MLflow AI Gateway.

        Args:
            endpoint: The name of the endpoint to retrieve.

        Returns:
            An `Endpoint` object representing the endpoint.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("http://localhost:5000")
            endpoint = client.get_endpoint(endpoint="chat")
            assert endpoint.dict() == {
                "name": "chat",
                "endpoint_type": "llm/v1/chat",
                "model": {"name": "gpt-4o-mini", "provider": "openai"},
                "endpoint_url": "http://localhost:5000/gateway/chat/invocations",
            }
        """
        # Delayed import to avoid importing mlflow.gateway in the module scope
        from mlflow.deployments.server.config import Endpoint

        route = join_paths(MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, endpoint)
        response = self._call_endpoint("GET", route)
        return Endpoint(
            **{
                **response,
                "endpoint_url": resolve_endpoint_url(self.target_uri, response["endpoint_url"]),
            }
        )

    def _list_endpoints(self, page_token=None) -> "PagedList[Endpoint]":
        # Delayed import to avoid importing mlflow.gateway in the module scope
        from mlflow.deployments.server.config import Endpoint

        params = None if page_token is None else {"page_token": page_token}
        response_json = self._call_endpoint(
            "GET", MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, json_body=params
        )
        routes = [
            Endpoint(
                **{
                    **resp,
                    "endpoint_url": resolve_endpoint_url(
                        self.target_uri,
                        resp["endpoint_url"],
                    ),
                }
            )
            for resp in response_json.get("endpoints", [])
        ]
        next_page_token = response_json.get("next_page_token")
        return PagedList(routes, next_page_token)

    @experimental
    def list_endpoints(self) -> "List[Endpoint]":
        """
        List endpoints configured for the MLflow AI Gateway.

        Returns:
            A list of ``Endpoint`` objects.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("http://localhost:5000")

            endpoints = client.list_endpoints()
            assert [e.dict() for e in endpoints] == [
                {
                    "name": "chat",
                    "endpoint_type": "llm/v1/chat",
                    "model": {"name": "gpt-4o-mini", "provider": "openai"},
                    "endpoint_url": "http://localhost:5000/gateway/chat/invocations",
                },
            ]

        """
        endpoints = []
        next_page_token = None
        while True:
            page = self._list_endpoints(next_page_token)
            endpoints.extend(page)
            next_page_token = page.token
            if next_page_token is None:
                break
        return endpoints

    @experimental
    def predict(self, deployment_name=None, inputs=None, endpoint=None) -> Dict[str, Any]:
        """
        Submit a query to a configured provider endpoint.

        Args:
            deployment_name: Unused.
            inputs: The inputs to the query, as a dictionary.
            endpoint: The name of the endpoint to query.

        Returns:
            A dictionary containing the response from the endpoint.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("http://localhost:5000")

            response = client.predict(
                endpoint="chat",
                inputs={"messages": [{"role": "user", "content": "Hello"}]},
            )
            assert response == {
                "id": "chatcmpl-8OLoQuaeJSLybq3NBoe0w5eyqjGb9",
                "object": "chat.completion",
                "created": 1700814410,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! How can I assist you today?",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 9,
                    "total_tokens": 18,
                },
            }

        Additional parameters that are valid for a given provider and endpoint configuration can be
        included with the request as shown below, using an openai completions endpoint request as
        an example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("http://localhost:5000")
            client.predict(
                endpoint="completions",
                inputs={
                    "prompt": "Hello!",
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
            )
        """
        query_route = join_paths(
            MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE, endpoint, MLFLOW_DEPLOYMENTS_QUERY_SUFFIX
        )
        try:
            return self._call_endpoint(
                "POST", query_route, inputs, MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT.get()
            )
        except MlflowException as e:
            if isinstance(e.__cause__, requests.exceptions.Timeout):
                raise MlflowException(
                    message=(
                        "The provider has timed out while generating a response to your "
                        "query. Please evaluate the available parameters for the query "
                        "that you are submitting. Some parameter values and inputs can "
                        "increase the computation time beyond the allowable route "
                        f"timeout of {MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT} "
                        "seconds."
                    ),
                    error_code=BAD_REQUEST,
                )
            raise e


def run_local(name, model_uri, flavor=None, config=None):
    pass


def target_help():
    pass
