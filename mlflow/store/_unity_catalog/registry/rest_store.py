import functools
import logging
import tempfile

from mlflow.protos.service_pb2 import GetRun, MlflowService
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
    TemporaryCredentials,
    MODEL_VERSION_OPERATION_READ_WRITE,
)
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.store.entities.paged_list import PagedList
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    extract_api_info_for_service,
    extract_all_api_info_for_service,
    _REST_API_PATH_PREFIX,
    verify_rest_response,
    http_request,
)
from mlflow.store._unity_catalog.registry.utils import get_artifact_repo_from_storage_info
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.store._unity_catalog.registry.utils import (
    model_version_from_uc_proto,
    registered_model_from_uc_proto,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds


_DATABRICKS_ORG_ID_HEADER = "x-databricks-org-id"
_TRACKING_METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _REST_API_PATH_PREFIX)
_METHOD_TO_INFO = extract_api_info_for_service(UcModelRegistryService, _REST_API_PATH_PREFIX)
_METHOD_TO_ALL_INFO = extract_all_api_info_for_service(
    UcModelRegistryService, _REST_API_PATH_PREFIX
)

_logger = logging.getLogger(__name__)


def _require_arg_unspecified(arg_name, arg_value, default_values=None, message=None):
    default_values = [None] if default_values is None else default_values
    if arg_value not in default_values:
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

    :param store_uri: URI with scheme 'databricks-uc'
    :param tracking_uri: URI of the Databricks MLflow tracking server from which to fetch
                         run info and download run artifacts, when creating new model
                         versions from source artifacts logged to an MLflow run.
    """

    def __init__(self, store_uri, tracking_uri):
        super().__init__(get_host_creds=functools.partial(get_databricks_host_creds, store_uri))
        self.tracking_uri = tracking_uri
        self.get_tracking_host_creds = functools.partial(get_databricks_host_creds, tracking_uri)

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
            GetRun: GetRun.Response,
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
        _require_arg_unspecified(arg_name="tags", arg_value=tags, default_values=[[], None])
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
    def _finalize_model_version(self, name, version):
        """
        Finalize a UC model version after its files have been written to managed storage,
        updating its status from PENDING_REGISTRATION to READY
        :param name: Registered model name
        :param version: Model version number
        :return Protobuf ModelVersion describing the finalized model version
        """
        req_body = message_to_json(FinalizeModelVersionRequest(name=name, version=version))
        return self._call_endpoint(FinalizeModelVersionRequest, req_body).model_version

    def _get_temporary_model_version_write_credentials(self, name, version) -> TemporaryCredentials:
        """
        Get temporary credentials for uploading model version files
        :param name: Registered model name
        :param version: Model version number
        :return: mlflow.protos.databricks_uc_registry_messages_pb2.TemporaryCredentials
                 containing temporary model version credentials
        """
        req_body = message_to_json(
            GenerateTemporaryModelVersionCredentialsRequest(
                name=name, version=version, operation=MODEL_VERSION_OPERATION_READ_WRITE
            )
        )
        return self._call_endpoint(
            GenerateTemporaryModelVersionCredentialsRequest, req_body
        ).credentials

    def _get_workspace_id(self, run_id):
        if run_id is None:
            return None
        host_creds = self.get_tracking_host_creds()
        endpoint, method = _TRACKING_METHOD_TO_INFO[GetRun]
        response = http_request(
            host_creds=host_creds, endpoint=endpoint, method=method, params={"run_id": run_id}
        )
        response = verify_rest_response(response, endpoint)
        if _DATABRICKS_ORG_ID_HEADER not in response.headers:
            _logger.warning(
                "Unable to get model version source run's workspace ID from request headers. "
                "No run link will be recorded for the model version"
            )
            return None
        return response.headers[_DATABRICKS_ORG_ID_HEADER]

    def _validate_model_signature(self, local_model_dir):
        # Import Model here instead of in the top level, to avoid circular import; the
        # mlflow.models.model module imports from MLflow tracking, which triggers an import of
        # this file during store registry initialization
        from mlflow.models.model import Model

        try:
            model = Model.load(local_model_dir)
        except Exception as e:
            raise MlflowException(
                "Unable to load model metadata. Ensure the source path of the model "
                "being registered points to a valid MLflow model directory "
                "(see https://mlflow.org/docs/latest/models.html#storage-format) containing a "
                "model signature (https://mlflow.org/docs/latest/models.html#model-signature) "
                "specifying both input and output type specifications."
            ) from e
        signature_required_explanation = (
            "All models in the Unity Catalog must be logged with a "
            "model signature containing both input and output "
            "type specifications. See "
            "https://mlflow.org/docs/latest/models.html#model-signature "
            "for details on how to log a model with a signature"
        )
        if model.signature is None:
            raise MlflowException(
                "Model passed for registration did not contain any signature metadata. "
                f"{signature_required_explanation}"
            )
        if model.signature.outputs is None:
            raise MlflowException(
                "Model passed for registration contained a signature that includes only inputs. "
                f"{signature_required_explanation}"
            )

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
        _require_arg_unspecified(arg_name="tags", arg_value=tags, default_values=[[], None])
        source_workspace_id = self._get_workspace_id(run_id)
        req_body = message_to_json(
            CreateModelVersionRequest(
                name=name,
                source=source,
                run_id=run_id,
                description=description,
                run_tracking_server_id=source_workspace_id,
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_model_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=source, dst_path=tmpdir, tracking_uri=self.tracking_uri
            )
            self._validate_model_signature(local_model_dir)
            model_version = self._call_endpoint(CreateModelVersionRequest, req_body).model_version
            version_number = model_version.version
            scoped_token = self._get_temporary_model_version_write_credentials(
                name=name, version=version_number
            )
            store = get_artifact_repo_from_storage_info(
                storage_location=model_version.storage_location, scoped_token=scoped_token
            )
            store.log_artifacts(local_dir=local_model_dir, artifact_path="")
        finalized_mv = self._finalize_model_version(name=name, version=version_number)
        return model_version_from_uc_proto(finalized_mv)

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

    def set_registered_model_alias(self, name, alias, version):
        """
        Set a registered model alias pointing to a model version.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :param version: Registered model version number.
        :return: None
        """
        _raise_unsupported_method(method="set_registered_model_alias")

    def delete_registered_model_alias(self, name, alias):
        """
        Delete an alias associated with a registered model.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :return: None
        """
        _raise_unsupported_method(method="delete_registered_model_alias")

    def get_model_version_by_alias(self, name, alias):
        """
        Get the model version instance by name and alias.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        _raise_unsupported_method(method="get_model_version_by_alias")
