from abc import ABCMeta, abstractmethod

from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.annotations import experimental
from mlflow.utils.rest_utils import (
    call_endpoint,
    call_endpoints,
)


@experimental
class BaseRestStore(AbstractStore, metaclass=ABCMeta):  # pylint: disable=abstract-method
    """
    Base class client for a remote model registry server accessed via REST API calls
    """

    def __init__(self, get_host_creds):
        super().__init__()
        self.get_host_creds = get_host_creds

    @abstractmethod
    def _get_all_endpoints_from_method(self, method):
        pass

    @abstractmethod
    def _get_endpoint_from_method(self, method):
        pass

    @abstractmethod
    def _get_response_from_method(self, method):
        pass

    def _call_endpoint(self, api, json_body, call_all_endpoints=False, extra_headers=None):
        response_proto = self._get_response_from_method(api)
        if call_all_endpoints:
            endpoints = self._get_all_endpoints_from_method(api)
            return call_endpoints(
                self.get_host_creds(), endpoints, json_body, response_proto, extra_headers
            )
        else:
            endpoint, method = self._get_endpoint_from_method(api)
            return call_endpoint(
                self.get_host_creds(), endpoint, method, json_body, response_proto, extra_headers
            )
