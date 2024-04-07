import json
import posixpath
from typing import Any, Dict, Iterator, Optional

from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments.constants import (
    MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
)
from mlflow.environment_variables import (
    MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT,
    MLFLOW_HTTP_REQUEST_TIMEOUT,
)
from mlflow.exceptions import MlflowException
from mlflow.utils import AttrDict
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request


class DatabricksEndpoint(AttrDict):
    """
    A dictionary-like object representing a Databricks serving endpoint.

    .. code-block:: python

        endpoint = DatabricksEndpoint(
            {
                "name": "chat",
                "creator": "alice@company.com",
                "creation_timestamp": 0,
                "last_updated_timestamp": 0,
                "state": {...},
                "config": {...},
                "tags": [...],
                "id": "88fd3f75a0d24b0380ddc40484d7a31b",
            }
        )
        assert endpoint.name == "chat"
    """

    pass


@experimental
class DatabricksDeploymentClient(BaseDeploymentClient):
    """
    Client for interacting with Databricks serving endpoints.

    Example:

    First, set up credentials for authentication:

    .. code-block:: bash

        export DATABRICKS_HOST=...
        export DATABRICKS_TOKEN=...

    .. seealso::

        See https://docs.databricks.com/en/dev-tools/auth.html for other authentication methods.

    Then, create a deployment client and use it to interact with Databricks serving endpoints:

    .. code-block:: python

        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("databricks")
        endpoints = client.list_endpoints()
        assert endpoints == [
            {
                "name": "chat",
                "creator": "alice@company.com",
                "creation_timestamp": 0,
                "last_updated_timestamp": 0,
                "state": {...},
                "config": {...},
                "tags": [...],
                "id": "88fd3f75a0d24b0380ddc40484d7a31b",
            },
        ]
    """

    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        .. warning::

            This method is not implemented for `DatabricksDeploymentClient`.
        """
        raise NotImplementedError

    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        """
        .. warning::

            This method is not implemented for `DatabricksDeploymentClient`.
        """
        raise NotImplementedError

    def delete_deployment(self, name, config=None, endpoint=None):
        """
        .. warning::

            This method is not implemented for `DatabricksDeploymentClient`.
        """
        raise NotImplementedError

    def list_deployments(self, endpoint=None):
        """
        .. warning::

            This method is not implemented for `DatabricksDeploymentClient`.
        """
        raise NotImplementedError

    def get_deployment(self, name, endpoint=None):
        """
        .. warning::

            This method is not implemented for `DatabricksDeploymentClient`.
        """
        raise NotImplementedError

    def _call_endpoint(
        self,
        *,
        method: str,
        prefix: str = "/api/2.0",
        route: Optional[str] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
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
            timeout=MLFLOW_HTTP_REQUEST_TIMEOUT.get() if timeout is None else timeout,
            raise_on_status=False,
            retry_codes=MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
            extra_headers={"X-Databricks-Endpoints-API-Client": "Databricks Deployment Client"},
            **call_kwargs,
        )
        augmented_raise_for_status(response)
        return DatabricksEndpoint(response.json())

    def _call_endpoint_stream(
        self,
        *,
        method: str,
        prefix: str = "/api/2.0",
        route: Optional[str] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> Iterator[str]:
        call_kwargs = {}
        if method.lower() == "get":
            call_kwargs["params"] = json_body
        else:
            call_kwargs["json"] = json_body

        response = http_request(
            host_creds=get_databricks_host_creds(self.target_uri),
            endpoint=posixpath.join(prefix, "serving-endpoints", route or ""),
            method=method,
            timeout=MLFLOW_HTTP_REQUEST_TIMEOUT.get() if timeout is None else timeout,
            raise_on_status=False,
            retry_codes=MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
            extra_headers={"X-Databricks-Endpoints-API-Client": "Databricks Deployment Client"},
            stream=True,  # Receive response content in streaming way.
            **call_kwargs,
        )
        augmented_raise_for_status(response)

        # Streaming response content are composed of multiple lines.
        # Each line format depends on specific endpoint
        return (
            line.strip()
            for line in response.iter_lines(decode_unicode=True)
            if line.strip()  # filter out keep-alive new lines
        )

    @experimental
    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        """
        Query a serving endpoint with the provided model inputs.
        See https://docs.databricks.com/api/workspace/servingendpoints/query for request/response
        schema.

        Args:
            deployment_name: Unused.
            inputs: A dictionary containing the model inputs to query.
            endpoint: The name of the serving endpoint to query.

        Returns:
            A :py:class:`DatabricksEndpoint` object containing the query response.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("databricks")
            response = client.predict(
                endpoint="chat",
                inputs={
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                    ],
                },
            )
            assert response == {
                "id": "chatcmpl-8OLm5kfqBAJD8CpsMANESWKpLSLXY",
                "object": "chat.completion",
                "created": 1700814265,
                "model": "gpt-4-0613",
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
        """
        return self._call_endpoint(
            method="POST",
            prefix="/",
            route=posixpath.join(endpoint, "invocations"),
            json_body=inputs,
            timeout=MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT.get(),
        )

    @experimental
    def predict_stream(
        self, deployment_name=None, inputs=None, endpoint=None
    ) -> Iterator[Dict[str, Any]]:
        """
        Submit a query to a configured provider endpoint, and get streaming response

        Args:
            deployment_name: Unused.
            inputs: The inputs to the query, as a dictionary.
            endpoint: The name of the endpoint to query.

        Returns:
            An iterator of dictionary containing the response from the endpoint.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("databricks")
            chunk_iter = client.predict_stream(
                endpoint="databricks-llama-2-70b-chat",
                inputs={
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "temperature": 0.0,
                    "n": 1,
                    "max_tokens": 500,
                },
            )
            for chunk in chunk_iter:
                print(chunk)
                # Example:
                # {
                #     "id": "82a834f5-089d-4fc0-ad6c-db5c7d6a6129",
                #     "object": "chat.completion.chunk",
                #     "created": 1712133837,
                #     "model": "llama-2-70b-chat-030424",
                #     "choices": [
                #         {
                #             "index": 0, "delta": {"role": "assistant", "content": "Hello"},
                #             "finish_reason": None,
                #         }
                #     ],
                #     "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
                # }
        """
        inputs = inputs or {}

        # Add stream=True param in request body to get streaming response
        # See https://docs.databricks.com/api/workspace/servingendpoints/query#stream
        chunk_line_iter = self._call_endpoint_stream(
            method="POST",
            prefix="/",
            route=posixpath.join(endpoint, "invocations"),
            json_body={**inputs, "stream": True},
            timeout=MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT.get(),
        )

        for line in chunk_line_iter:
            splits = line.split(":", 1)
            if len(splits) < 2:
                raise MlflowException(f"Unknown streaming response format: '{line}'.")
            key, value = splits
            if key != "data":
                raise MlflowException(f"Unknown streaming response format with key '{key}'.")

            value = value.strip()
            if value == "[DONE]":
                # Databricks endpoint streaming response ends with
                # a line of "data: [DONE]"
                return

            yield json.loads(value)

    @experimental
    def create_endpoint(self, name, config=None):
        """
        Create a new serving endpoint with the provided name and configuration.

        See https://docs.databricks.com/api/workspace/servingendpoints/create for request/response
        schema.

        Args:
            name: The name of the serving endpoint to create.
            config: A dictionary containing the configuration of the serving endpoint to create.

        Returns:
            A :py:class:`DatabricksEndpoint` object containing the request response.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("databricks")
            endpoint = client.create_endpoint(
                name="chat",
                config={
                    "served_entities": [
                        {
                            "name": "test",
                            "external_model": {
                                "name": "gpt-4",
                                "provider": "openai",
                                "task": "llm/v1/chat",
                                "openai_config": {
                                    "openai_api_key": "{{secrets/scope/key}}",
                                },
                            },
                        }
                    ],
                },
            )
            assert endpoint == {
                "name": "chat",
                "creator": "alice@company.com",
                "creation_timestamp": 0,
                "last_updated_timestamp": 0,
                "state": {...},
                "config": {...},
                "tags": [...],
                "id": "88fd3f75a0d24b0380ddc40484d7a31b",
            }

        """
        config = config.copy() if config else {}  # avoid mutating config
        extras = {}
        for key in ("tags", "rate_limits"):
            if tags := config.pop(key, None):
                extras[key] = tags
        payload = {"name": name, "config": config, **extras}
        return self._call_endpoint(method="POST", json_body=payload)

    @experimental
    def update_endpoint(self, endpoint, config=None):
        """
        Update a specified serving endpoint with the provided configuration.
        See https://docs.databricks.com/api/workspace/servingendpoints/updateconfig for
        request/response schema.

        Args:
            endpoint: The name of the serving endpoint to update.
            config: A dictionary containing the configuration of the serving endpoint to update.

        Returns:
            A :py:class:`DatabricksEndpoint` object containing the request response.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("databricks")
            endpoint = client.update_endpoint(
                endpoint="chat",
                config={
                    "served_entities": [
                        {
                            "name": "test",
                            "external_model": {
                                "name": "gpt-4",
                                "provider": "openai",
                                "task": "llm/v1/chat",
                                "openai_config": {
                                    "openai_api_key": "{{secrets/scope/key}}",
                                },
                            },
                        }
                    ],
                },
            )
            assert endpoint == {
                "name": "chat",
                "creator": "alice@company.com",
                "creation_timestamp": 0,
                "last_updated_timestamp": 0,
                "state": {...},
                "config": {...},
                "tags": [...],
                "id": "88fd3f75a0d24b0380ddc40484d7a31b",
            }

            rate_limits = client.update_endpoint(
                endpoint="chat",
                config={
                    "rate_limits": [
                        {
                            "key": "user",
                            "renewal_period": "minute",
                            "calls": 10,
                        }
                    ],
                },
            )
            assert rate_limits == {
                "rate_limits": [
                    {
                        "key": "user",
                        "renewal_period": "minute",
                        "calls": 10,
                    }
                ],
            }
        """
        if list(config) == ["rate_limits"]:
            return self._call_endpoint(
                method="PUT", route=posixpath.join(endpoint, "rate-limits"), json_body=config
            )
        else:
            return self._call_endpoint(
                method="PUT", route=posixpath.join(endpoint, "config"), json_body=config
            )

    @experimental
    def delete_endpoint(self, endpoint):
        """
        Delete a specified serving endpoint.
        See https://docs.databricks.com/api/workspace/servingendpoints/delete for request/response
        schema.

        Args:
            endpoint: The name of the serving endpoint to delete.

        Returns:
            A DatabricksEndpoint object containing the request response.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("databricks")
            client.delete_endpoint(endpoint="chat")
        """
        return self._call_endpoint(method="DELETE", route=endpoint)

    @experimental
    def list_endpoints(self):
        """
        Retrieve all serving endpoints.

        See https://docs.databricks.com/api/workspace/servingendpoints/list for request/response
        schema.

        Returns:
            A list of :py:class:`DatabricksEndpoint` objects containing the request response.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("databricks")
            endpoints = client.list_endpoints()
            assert endpoints == [
                {
                    "name": "chat",
                    "creator": "alice@company.com",
                    "creation_timestamp": 0,
                    "last_updated_timestamp": 0,
                    "state": {...},
                    "config": {...},
                    "tags": [...],
                    "id": "88fd3f75a0d24b0380ddc40484d7a31b",
                },
            ]

        """
        return self._call_endpoint(method="GET").endpoints

    @experimental
    def get_endpoint(self, endpoint):
        """
        Get a specified serving endpoint.
        See https://docs.databricks.com/api/workspace/servingendpoints/get for request/response
        schema.

        Args:
            endpoint: The name of the serving endpoint to get.

        Returns:
            A DatabricksEndpoint object containing the request response.

        Example:

        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("databricks")
            endpoint = client.get_endpoint(endpoint="chat")
            assert endpoint == {
                "name": "chat",
                "creator": "alice@company.com",
                "creation_timestamp": 0,
                "last_updated_timestamp": 0,
                "state": {...},
                "config": {...},
                "tags": [...],
                "id": "88fd3f75a0d24b0380ddc40484d7a31b",
            }
        """
        return self._call_endpoint(method="GET", route=endpoint)


def run_local(name, model_uri, flavor=None, config=None):
    pass


def target_help():
    pass
