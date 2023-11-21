import json
from typing import Optional

import requests

from mlflow import MlflowException
from mlflow.deployments import BaseDeploymentClient
from mlflow.deployments.constants import (
    MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
    MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT,
)
from mlflow.deployments.server.config import Endpoint
from mlflow.deployments.server.constants import (
    MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE,
    MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE,
    MLFLOW_DEPLOYMENTS_QUERY_SUFFIX,
)
from mlflow.deployments.utils import assemble_uri_path, resolve_route_url
from mlflow.environment_variables import MLFLOW_HTTP_REQUEST_TIMEOUT
from mlflow.protos.databricks_pb2 import BAD_REQUEST
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking._tracking_service.utils import _get_default_host_creds
from mlflow.utils.annotations import experimental
from mlflow.utils.rest_utils import augmented_raise_for_status, http_request


@experimental
class MLflowDeploymentClient(BaseDeploymentClient):
    """
    TODO
    """

    def create_deployment(self, name, model_uri, flavor=None, config=None, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def update_deployment(self, name, model_uri=None, flavor=None, config=None, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def delete_deployment(self, name, config=None, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def list_deployments(self, endpoint=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def get_deployment(self, name, endpoint=None):
        """
        TODO
        """
        raise NotImplementedError

    def create_endpoint(self, name, config=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def update_endpoint(self, endpoint, config=None):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def delete_endpoint(self, endpoint):
        """
        .. warning::
            This method is not implemented for `MLflowDeploymentClient`.
        """
        raise NotImplementedError

    def _call_endpoint(
        self,
        method: str,
        route: str,
        json_body: Optional[str] = None,
        timeout: int = MLFLOW_HTTP_REQUEST_TIMEOUT.get(),
    ):
        call_kwargs = {}
        if method.lower() == "get":
            call_kwargs["params"] = json_body
        else:
            call_kwargs["json"] = json_body

        response = http_request(
            host_creds=_get_default_host_creds(self.target_uri),
            endpoint=route,
            method=method,
            timeout=timeout,
            retry_codes=MLFLOW_DEPLOYMENT_CLIENT_REQUEST_RETRY_CODES,
            raise_on_status=False,
            **call_kwargs,
        )
        augmented_raise_for_status(response)
        return response.json()

    @experimental
    def get_endpoint(self, endpoint):
        """
        TODO
        """
        route = assemble_uri_path([MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, endpoint])
        response = self._call_endpoint("GET", route)
        response["endpoint_url"] = resolve_route_url(self.target_uri, response["endpoint_url"])
        return Endpoint(**response)

    @experimental
    def list_endpoints(self, page_token=None):
        """
        TODO
        """
        params = {"page_token": page_token} if page_token is not None else None
        response_json = self._call_endpoint(
            "GET", MLFLOW_DEPLOYMENTS_CRUD_ENDPOINT_BASE, json_body=params
        )
        routes = [
            Endpoint(
                **{
                    **resp,
                    "endpoint_url": resolve_route_url(
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
    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        """
        TODO
        """
        query_route = assemble_uri_path(
            [MLFLOW_DEPLOYMENTS_ENDPOINTS_BASE, endpoint, MLFLOW_DEPLOYMENTS_QUERY_SUFFIX]
        )
        try:
            return self._call_endpoint(
                "POST", query_route, inputs, MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT.get()
            )
        except MlflowException as e:
            if isinstance(e.__cause__, requests.exceptions.Timeout):
                timeout_message = (
                    "The provider has timed out while generating a response to your "
                    "query. Please evaluate the available parameters for the query "
                    "that you are submitting. Some parameter values and inputs can "
                    "increase the computation time beyond the allowable route "
                    f"timeout of {MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT} "
                    "seconds."
                )
                raise MlflowException(message=timeout_message, error_code=BAD_REQUEST)
            else:
                raise e


def run_local(name, model_uri, flavor=None, config=None):
    pass


def target_help():
    pass
