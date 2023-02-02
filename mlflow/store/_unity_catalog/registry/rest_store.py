import logging

from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    CreateRegisteredModelRequest,
    CreateRegisteredModelResponse,
    UpdateRegisteredModelRequest,
    UpdateRegisteredModelResponse,
    DeleteRegisteredModelRequest,
    DeleteRegisteredModelResponse,
    CreateModelVersionRequest,
    CreateModelVersionResponse,
    FinalizeModelVersionRequest,
    FinalizeModelVersionResponse,
    UpdateModelVersionRequest,
    UpdateModelVersionResponse,
    DeleteModelVersionRequest,
    DeleteModelVersionResponse,
    GetModelVersionDownloadUriRequest,
    GetModelVersionDownloadUriResponse,
    SearchModelVersionsRequest,
    SearchModelVersionsResponse,
    GetRegisteredModelRequest,
    GetRegisteredModelResponse,
    GetModelVersionRequest,
    GetModelVersionResponse,
    SearchRegisteredModelsRequest,
    SearchRegisteredModelsResponse,
    GenerateTemporaryModelVersionCredentialsRequest,
    GenerateTemporaryModelVersionCredentialsResponse,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    extract_api_info_for_service,
    extract_all_api_info_for_service,
    _REST_API_PATH_PREFIX,
)
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.store._unity_catalog.registry.utils import (
    registered_model_from_uc_proto,
)

_METHOD_TO_INFO = extract_api_info_for_service(UcModelRegistryService, _REST_API_PATH_PREFIX)
_METHOD_TO_ALL_INFO = extract_all_api_info_for_service(
    UcModelRegistryService, _REST_API_PATH_PREFIX
)

_logger = logging.getLogger(__name__)


def _require_arg_unspecified(arg_name, arg_value, default_value=None, message=None):
    if arg_value != default_value:
        _raise_unsupported_arg(arg_name, message)


def _raise_unsupported_arg(arg_name, message=None):
    messages = [
        f"Argument {arg_name} is unsupported for models in the Unity Catalog.",
    ]
    if message is not None:
        messages.append(message)
    messages.append("See the user guide for more information")
    raise MlflowException(" ".join(messages))


def _raise_unsupported_method(method, message=None):
    messages = [
        f"Method {method} is unsupported for models in the Unity Catalog.",
    ]
    if message is not None:
        messages.append(message)
    messages.append("See the user guide for more information")
    raise MlflowException(" ".join(messages))


def _get_response_from_method(method):
    method_to_response = {
        CreateRegisteredModelRequest: CreateRegisteredModelResponse,
        UpdateRegisteredModelRequest: UpdateRegisteredModelResponse,
        DeleteRegisteredModelRequest: DeleteRegisteredModelResponse,
        CreateModelVersionRequest: CreateModelVersionResponse,
        FinalizeModelVersionRequest: FinalizeModelVersionResponse,
        UpdateModelVersionRequest: UpdateModelVersionResponse,
        DeleteModelVersionRequest: DeleteModelVersionResponse,
        GetModelVersionDownloadUriRequest: GetModelVersionDownloadUriResponse,
        SearchModelVersionsRequest: SearchModelVersionsResponse,
        GetRegisteredModelRequest: GetRegisteredModelResponse,
        GetModelVersionRequest: GetModelVersionResponse,
        SearchRegisteredModelsRequest: SearchRegisteredModelsResponse,
        # pylint: disable=line-too-long
        GenerateTemporaryModelVersionCredentialsRequest: GenerateTemporaryModelVersionCredentialsResponse,
    }
    return method_to_response[method]()


def _get_all_endpoints_from_method(method):
    return _METHOD_TO_ALL_INFO[method]


def _get_endpoint_from_method(method):
    return _METHOD_TO_INFO[method]


# TODO: re-enable the abstract-method check in a follow-up PR. It's disabled for now because
# we haven't yet implemented required model version CRUD APIs in this class
class UcModelRegistryStore(BaseRestStore):  # pylint: disable=abstract-method
    """
    Note:: Experimental: This entity may change or be removed in a future release without warning.
    Client for a remote model registry server accessed via REST API calls

    :param get_host_creds: Method to be invoked prior to every REST request to get the
      :py:class:`mlflow.rest_utils.MlflowHostCreds` for the request. Note that this
      is a function so that we can obtain fresh credentials in the case of expiry.
    """

    def __init__(self, get_host_creds):
        super().__init__(
            get_host_creds,
            get_response_from_method=_get_response_from_method,
            get_endpoint_from_method=_get_endpoint_from_method,
            get_all_endpoints_from_method=_get_all_endpoints_from_method,
        )

    # CRUD API for RegisteredModel objects

    def create_registered_model(self, name, tags=None, description=None):
        """
        Create a new registered model in backend store.

        :param name: Name of the new model. This is expected to be unique in the backend store.
        :param tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                     instances associated with this registered model.
        :param description: Description of the model.
        :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
                 created in the backend.
        """
        req_body = message_to_json(CreateRegisteredModelRequest(name=name, description=description))
        response_proto = self._call_endpoint(CreateRegisteredModelRequest, req_body)
        return registered_model_from_uc_proto(response_proto.registered_model)

    def update_registered_model(self, name, description):
        """
        Update description of the registered model.

        :param name: Registered model name.
        :param description: New description.
        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        req_body = message_to_json(UpdateRegisteredModelRequest(name=name, description=description))
        response_proto = self._call_endpoint(UpdateRegisteredModelRequest, req_body)
        return registered_model_from_uc_proto(response_proto.registered_model)

    def rename_registered_model(self, name, new_name):
        """
        Rename the registered model.

        :param name: Registered model name.
        :param new_name: New proposed name.
        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        _raise_unsupported_method(
            method="rename_registered_model",
            message="Use the Unity Catalog REST API to rename registered models",
        )

    def delete_registered_model(self, name):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param name: Registered model name.
        :return: None
        """
        req_body = message_to_json(DeleteRegisteredModelRequest(name=name))
        self._call_endpoint(DeleteRegisteredModelRequest, req_body)

    def search_registered_models(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for registered models in backend that satisfy the filter criteria.

        :param filter_string: Filter query string, defaults to searching all registered models.
        :param max_results: Maximum number of registered models desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_registered_models`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions. The pagination token for the next page can be
                obtained via the ``token`` attribute of the object.
        """
        _require_arg_unspecified("filter_string", filter_string)
        _require_arg_unspecified("order_by", order_by)
        req_body = message_to_json(
            SearchRegisteredModelsRequest(
                max_results=max_results,
                page_token=page_token,
            )
        )
        response_proto = self._call_endpoint(SearchRegisteredModelsRequest, req_body)
        registered_models = [
            registered_model_from_uc_proto(registered_model)
            for registered_model in response_proto.registered_models
        ]
        return PagedList(registered_models, response_proto.next_page_token)

    def get_registered_model(self, name):
        """
        Get registered model instance by name.

        :param name: Registered model name.
        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        req_body = message_to_json(GetRegisteredModelRequest(name=name))
        response_proto = self._call_endpoint(GetRegisteredModelRequest, req_body)
        return registered_model_from_uc_proto(response_proto.registered_model)

    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        :param name: Registered model name.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       each stage.
        :return: List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        _raise_unsupported_method(method="get_latest_versions")

    def set_registered_model_tag(self, name, tag):
        """
        Set a tag for the registered model.

        :param name: Registered model name.
        :param tag: :py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.
        :return: None
        """
        _raise_unsupported_method(method="set_registered_model_tag")

    def delete_registered_model_tag(self, name, key):
        """
        Delete a tag associated with the registered model.

        :param name: Registered model name.
        :param key: Registered model tag key.
        :return: None
        """
        _raise_unsupported_method(method="delete_registered_model_tag")
