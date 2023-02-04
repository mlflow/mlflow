import logging

from abc import ABCMeta

from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.annotations import experimental
from mlflow.utils.rest_utils import (
    call_endpoint,
    call_endpoints,
)

_logger = logging.getLogger(__name__)


def _get_response_from_method(method):
    return method.Response()


@experimental
class BaseRestStore(AbstractStore):  # pylint: disable=abstract-method
    """
    Note:: Experimental: This entity may change or be removed in a future release without warning.
    Base class client for a remote model registry server accessed via REST API calls
    """

    __metaclass__ = ABCMeta

    def __init__(
        self,
        get_host_creds,
        get_response_from_method,
        get_all_endpoints_from_method,
        get_endpoint_from_method,
    ):
        super().__init__()
        self._get_all_endpoints_from_method = get_all_endpoints_from_method
        self._get_endpoint_from_method = get_endpoint_from_method
        self._get_response_from_method = get_response_from_method
        self.get_host_creds = get_host_creds

    def _call_endpoint(self, api, json_body, call_all_endpoints=False):
        response_proto = self._get_response_from_method(api)
        if call_all_endpoints:
            endpoints = self._get_all_endpoints_from_method(api)
            return call_endpoints(self.get_host_creds(), endpoints, json_body, response_proto)
        else:
            endpoint, method = self._get_endpoint_from_method(api)
            return call_endpoint(self.get_host_creds(), endpoint, method, json_body, response_proto)
