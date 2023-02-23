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
    model_version_from_uc_proto,
    registered_model_from_uc_proto,
)
from mlflow.utils.annotations import experimental

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
        f"Argument '{arg_name}' is unsupported for models in the Unity Catalog.",
    ]
    if message is not None:
        messages.append(message)
    messages.append("See the user guide for more information")
    raise MlflowException(" ".join(messages))


def _raise_unsupported_method(method, message=None):
    messages = [
        f"Method '{method}' is unsupported for models in the Unity Catalog.",
    ]
    if message is not None:
        messages.append(message)
    messages.append("See the user guide for more information")
    raise MlflowException(" ".join(messages))


@experimental
class UcModelRegistryStore(BaseRestStore):
    """
    Client for a remote model registry server accessed via REST API calls

    :param get_host_creds: Method to be invoked prior to every REST request to get the
      :py:class:`mlflow.rest_utils.MlflowHostCreds` for the request. Note that this
      is a function so that we can obtain fresh credentials in the case of expiry.
    """

    def _get_response_from_method(self, method):
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

    def _get_endpoint_from_method(self, method):
        return _METHOD_TO_INFO[method]

    def _get_all_endpoints_from_method(self, method):
        return _METHOD_TO_ALL_INFO[method]

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
        _require_arg_unspecified("tags", tags)
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
        _raise_unsupported_method(
            method="get_latest_versions",
            message="If seeing this error while attempting to "
            "load a models:/ URI of the form models:/<name>/<stage>, note that "
            "staged-based model URIs are unsupported for models in UC. Future "
            "MLflow Python client versions will include support for model "
            "aliases and alias-based 'models:/' URIs "
            "of the form models:/<name>@<alias> as an alternative.",
        )

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

    # CRUD API for ModelVersion objects
    def create_model_version(
        self, name, source, run_id=None, tags=None, run_link=None, description=None
    ):
        """
        Create a new model version from given source and run ID.

        :param name: Registered model name.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model.
        :param tags: A list of :py:class:`mlflow.entities.model_registry.ModelVersionTag`
                     instances associated with this model version.
        :param run_link: Link to the run from an MLflow tracking server that generated this model.
        :param description: Description of the version.
        :return: A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 created in the backend.
        """
        _require_arg_unspecified(arg_name="run_link", arg_value=run_link)
        _require_arg_unspecified(arg_name="tags", arg_value=tags)
        # TODO: Implement client-side model version upload and finalization logic here
        req_body = message_to_json(
            CreateModelVersionRequest(
                name=name,
                source=source,
                run_id=run_id,
                description=description,
            )
        )
        response_proto = self._call_endpoint(CreateModelVersionRequest, req_body)
        return response_proto.model_version

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        """
        Update model version stage.

        :param name: Registered model name.
        :param version: Registered model version.
        :param stage: New desired stage for this model version.
        :param archive_existing_versions: If this flag is set to ``True``, all existing model
            versions in the stage will be automatically moved to the "archived" stage. Only valid
            when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be raised.

        """
        _raise_unsupported_method(method="transition_model_version_stage")

    def update_model_version(self, name, version, description):
        """
        Update metadata associated with a model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.
        :param description: New model description.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        req_body = message_to_json(
            UpdateModelVersionRequest(name=name, version=str(version), description=description)
        )
        response_proto = self._call_endpoint(UpdateModelVersionRequest, req_body)
        return model_version_from_uc_proto(response_proto.model_version)

    def delete_model_version(self, name, version):
        """
        Delete model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.
        :return: None
        """
        req_body = message_to_json(DeleteModelVersionRequest(name=name, version=str(version)))
        self._call_endpoint(DeleteModelVersionRequest, req_body)

    def get_model_version(self, name, version):
        """
        Get the model version instance by name and version.

        :param name: Registered model name.
        :param version: Registered model version.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        req_body = message_to_json(GetModelVersionRequest(name=name, version=str(version)))
        response_proto = self._call_endpoint(GetModelVersionRequest, req_body)
        return model_version_from_uc_proto(response_proto.model_version)

    def get_model_version_download_uri(self, name, version):
        """
        Get the download location in Model Registry for this model version.
        NOTE: For first version of Model Registry, since the models are not copied over to another
              location, download URI points to input source path.

        :param name: Registered model name.
        :param version: Registered model version.
        :return: A single URI location that allows reads for downloading.
        """
        req_body = message_to_json(
            GetModelVersionDownloadUriRequest(name=name, version=str(version))
        )
        response_proto = self._call_endpoint(GetModelVersionDownloadUriRequest, req_body)
        return response_proto.artifact_uri

    def search_model_versions(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: A filter string expression. Currently supports a single filter
                              condition either name of model like ``name = 'model_name'`` or
                              ``run_id = '...'``.
        :param max_results: Maximum number of model versions desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_model_versions`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 objects that satisfy the search expressions. The pagination token for the next
                 page can be obtained via the ``token`` attribute of the object.
        """
        req_body = message_to_json(SearchModelVersionsRequest(filter=filter_string))
        response_proto = self._call_endpoint(SearchModelVersionsRequest, req_body)
        model_versions = [model_version_from_uc_proto(mvd) for mvd in response_proto.model_versions]
        return PagedList(model_versions, response_proto.next_page_token)

    def set_model_version_tag(self, name, version, tag):
        """
        Set a tag for the model version.

        :param name: Registered model name.
        :param version: Registered model version.
        :param tag: :py:class:`mlflow.entities.model_registry.ModelVersionTag` instance to log.
        """
        _raise_unsupported_method(method="set_model_version_tag")

    def delete_model_version_tag(self, name, version, key):
        """
        Delete a tag associated with the model version.

        :param name: Registered model name.
        :param version: Registered model version.
        :param key: Tag key.
        """
        _raise_unsupported_method(method="delete_model_version_tag")
