import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from time import sleep

from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.annotations import experimental
from mlflow.utils.rest_utils import (
    call_endpoint,
    call_endpoints,
)

_logger = logging.getLogger(__name__)

AWAIT_MODEL_VERSION_CREATE_SLEEP_DURATION_SECONDS = 3


@experimental
class BaseRestStore(AbstractStore):  # pylint: disable=abstract-method
    """
    Base class client for a remote model registry server accessed via REST API calls
    """

    __metaclass__ = ABCMeta

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

    def _await_model_version_creation_impl(self, name, version, await_creation_for, hint=""):
        _logger.info(
            f"Waiting up to {await_creation_for} seconds for model version to finish creation. "
            f"Model name: {name}, version {version}",
        )
        max_datetime = datetime.utcnow() + timedelta(seconds=await_creation_for)
        pending_status = ModelVersionStatus.to_string(ModelVersionStatus.PENDING_REGISTRATION)
        while mv.status == pending_status:
            if datetime.utcnow() > max_datetime:
                raise MlflowException(
                    f"Exceeded max wait time for model name: {mv.name} version: {mv.version} "
                    f"to become READY. Status: {mv.status} Wait Time: {await_creation_for}"
                    f".{hint}"
                )
            mv = self.get_model_version(mv.name, mv.version)
            sleep(AWAIT_MODEL_VERSION_CREATE_SLEEP_DURATION_SECONDS)
        if mv.status != ModelVersionStatus.to_string(ModelVersionStatus.READY):
            raise MlflowException(
                f"Model version creation failed for model name: {mv.name} version: "
                f"{mv.version} with status: {mv.status} and message: {mv.status_message}"
            )
