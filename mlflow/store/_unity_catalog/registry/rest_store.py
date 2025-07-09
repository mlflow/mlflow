import base64
import functools
import json
import logging
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union

import google.protobuf.empty_pb2

import mlflow
from mlflow.entities import Run
from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.model_registry.prompt import Prompt
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    MODEL_VERSION_OPERATION_READ_WRITE,
    CreateModelVersionRequest,
    CreateModelVersionResponse,
    CreateRegisteredModelRequest,
    CreateRegisteredModelResponse,
    DeleteModelVersionRequest,
    DeleteModelVersionResponse,
    DeleteModelVersionTagRequest,
    DeleteModelVersionTagResponse,
    DeleteRegisteredModelAliasRequest,
    DeleteRegisteredModelAliasResponse,
    DeleteRegisteredModelRequest,
    DeleteRegisteredModelResponse,
    DeleteRegisteredModelTagRequest,
    DeleteRegisteredModelTagResponse,
    Entity,
    FinalizeModelVersionRequest,
    FinalizeModelVersionResponse,
    GenerateTemporaryModelVersionCredentialsRequest,
    GenerateTemporaryModelVersionCredentialsResponse,
    GetModelVersionByAliasRequest,
    GetModelVersionByAliasResponse,
    GetModelVersionDownloadUriRequest,
    GetModelVersionDownloadUriResponse,
    GetModelVersionRequest,
    GetModelVersionResponse,
    GetRegisteredModelRequest,
    GetRegisteredModelResponse,
    Job,
    Lineage,
    LineageHeaderInfo,
    Notebook,
    SearchModelVersionsRequest,
    SearchModelVersionsResponse,
    SearchRegisteredModelsRequest,
    SearchRegisteredModelsResponse,
    Securable,
    SetModelVersionTagRequest,
    SetModelVersionTagResponse,
    SetRegisteredModelAliasRequest,
    SetRegisteredModelAliasResponse,
    SetRegisteredModelTagRequest,
    SetRegisteredModelTagResponse,
    StorageMode,
    Table,
    TemporaryCredentials,
    UpdateModelVersionRequest,
    UpdateModelVersionResponse,
    UpdateRegisteredModelRequest,
    UpdateRegisteredModelResponse,
)
from mlflow.protos.databricks_uc_registry_service_pb2 import UcModelRegistryService
from mlflow.protos.service_pb2 import GetRun, MlflowService
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    CreatePromptRequest,
    CreatePromptVersionRequest,
    DeletePromptAliasRequest,
    DeletePromptRequest,
    DeletePromptTagRequest,
    DeletePromptVersionRequest,
    DeletePromptVersionTagRequest,
    GetPromptRequest,
    GetPromptVersionByAliasRequest,
    GetPromptVersionRequest,
    LinkPromptsToTracesRequest,
    LinkPromptVersionsToModelsRequest,
    LinkPromptVersionsToRunsRequest,
    PromptVersionLinkEntry,
    SearchPromptsRequest,
    SearchPromptsResponse,
    SearchPromptVersionsRequest,
    SearchPromptVersionsResponse,
    SetPromptAliasRequest,
    SetPromptTagRequest,
    SetPromptVersionTagRequest,
    UnityCatalogSchema,
    UpdatePromptRequest,
    UpdatePromptVersionRequest,
)
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    Prompt as ProtoPrompt,
)
from mlflow.protos.unity_catalog_prompt_messages_pb2 import (
    PromptVersion as ProtoPromptVersion,
)
from mlflow.protos.unity_catalog_prompt_service_pb2 import UnityCatalogPromptService
from mlflow.store._unity_catalog.lineage.constants import (
    _DATABRICKS_LINEAGE_ID_HEADER,
    _DATABRICKS_ORG_ID_HEADER,
)
from mlflow.store._unity_catalog.registry.utils import (
    mlflow_tags_to_proto,
    mlflow_tags_to_proto_version_tags,
    proto_info_to_mlflow_prompt_info,
    proto_to_mlflow_prompt,
)
from mlflow.store.artifact.databricks_sdk_models_artifact_repo import (
    DatabricksSDKModelsArtifactRepository,
)
from mlflow.store.artifact.presigned_url_artifact_repo import (
    PresignedUrlArtifactRepository,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
    get_artifact_repo_from_storage_info,
    get_full_name_from_sc,
    is_databricks_sdk_models_artifact_repository_enabled,
    model_version_from_uc_proto,
    model_version_search_from_uc_proto,
    registered_model_from_uc_proto,
    registered_model_search_from_uc_proto,
    uc_model_version_tag_from_mlflow_tags,
    uc_registered_model_tag_from_mlflow_tags,
)
from mlflow.utils.databricks_utils import (
    _print_databricks_deployment_job_url,
    get_databricks_host_creds,
    is_databricks_uri,
)
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_JOB_RUN_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
)
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    call_endpoint,
    extract_all_api_info_for_service,
    extract_api_info_for_service,
    http_request,
    verify_rest_response,
)
from mlflow.utils.uri import is_fuse_or_uc_volumes_uri

_TRACKING_METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _REST_API_PATH_PREFIX)
_METHOD_TO_INFO = {
    **extract_api_info_for_service(UcModelRegistryService, _REST_API_PATH_PREFIX),
    **extract_api_info_for_service(UnityCatalogPromptService, _REST_API_PATH_PREFIX),
}
_METHOD_TO_ALL_INFO = {
    **extract_all_api_info_for_service(UcModelRegistryService, _REST_API_PATH_PREFIX),
    **extract_all_api_info_for_service(UnityCatalogPromptService, _REST_API_PATH_PREFIX),
}

_logger = logging.getLogger(__name__)
_DELTA_TABLE = "delta_table"
_MAX_LINEAGE_DATA_SOURCES = 10

# Pre-compiled regex patterns for better performance in search operations
_CATALOG_PATTERN = re.compile(r"catalog\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_SCHEMA_PATTERN = re.compile(r"schema\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)


@dataclass
class _CatalogSchemaFilter:
    """Internal class to hold parsed catalog, schema, and remaining filter."""

    catalog_name: str
    schema_name: str
    remaining_filter: Optional[str]


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
    raise MlflowException(" ".join(messages))


def _raise_unsupported_method(method, message=None):
    messages = [
        f"Method '{method}' is unsupported for models in the Unity Catalog.",
    ]
    if message is not None:
        messages.append(message)
    raise MlflowException(" ".join(messages))


def _load_model(local_model_dir):
    # Import Model here instead of in the top level, to avoid circular import; the
    # mlflow.models.model module imports from MLflow tracking, which triggers an import of
    # this file during store registry initialization
    from mlflow.models.model import Model

    try:
        return Model.load(local_model_dir)
    except Exception as e:
        raise MlflowException(
            "Unable to load model metadata. Ensure the source path of the model "
            "being registered points to a valid MLflow model directory "
            "(see https://mlflow.org/docs/latest/models.html#storage-format) containing a "
            "model signature (https://mlflow.org/docs/latest/models.html#model-signature) "
            "specifying both input and output type specifications."
        ) from e


def get_feature_dependencies(model_dir):
    """
    Gets the features which a model depends on. This functionality is only implemented on
    Databricks. In OSS mlflow, the dependencies are always empty ("").
    """
    model = _load_model(model_dir)
    model_info = model.get_model_info()
    if (
        model_info.flavors.get("python_function", {}).get("loader_module")
        == mlflow.models.model._DATABRICKS_FS_LOADER_MODULE
    ):
        raise MlflowException(
            "This model was packaged by Databricks Feature Store and can only be registered on a "
            "Databricks cluster."
        )
    return ""


def get_model_version_dependencies(model_dir):
    """
    Gets the specified dependencies for a particular model version and formats them
    to be passed into CreateModelVersion.
    """
    from mlflow.models.resources import ResourceType

    model = _load_model(model_dir)
    model_info = model.get_model_info()
    dependencies = []

    # Try to get model.auth_policy.system_auth_policy.resources. If that is not found or empty,
    # then use model.resources.
    if model.auth_policy:
        databricks_resources = model.auth_policy.get("system_auth_policy", {}).get("resources", {})
    else:
        databricks_resources = model.resources

    if databricks_resources:
        databricks_dependencies = databricks_resources.get("databricks", {})
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.VECTOR_SEARCH_INDEX.value,
                "DATABRICKS_VECTOR_INDEX",
            )
        )
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.SERVING_ENDPOINT.value,
                "DATABRICKS_MODEL_ENDPOINT",
            )
        )
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.FUNCTION.value,
                "DATABRICKS_UC_FUNCTION",
            )
        )
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.UC_CONNECTION.value,
                "DATABRICKS_UC_CONNECTION",
            )
        )
        dependencies.extend(
            _fetch_langchain_dependency_from_model_resources(
                databricks_dependencies,
                ResourceType.TABLE.value,
                "DATABRICKS_TABLE",
            )
        )
    else:
        # These types of dependencies are required for old models that didn't use
        # resources so they can be registered correctly to UC
        _DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY = "databricks_vector_search_index_name"
        _DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY = "databricks_embeddings_endpoint_name"
        _DATABRICKS_LLM_ENDPOINT_NAME_KEY = "databricks_llm_endpoint_name"
        _DATABRICKS_CHAT_ENDPOINT_NAME_KEY = "databricks_chat_endpoint_name"
        _DB_DEPENDENCY_KEY = "databricks_dependency"

        databricks_dependencies = model_info.flavors.get("langchain", {}).get(
            _DB_DEPENDENCY_KEY, {}
        )

        index_names = _fetch_langchain_dependency_from_model_info(
            databricks_dependencies, _DATABRICKS_VECTOR_SEARCH_INDEX_NAME_KEY
        )
        for index_name in index_names:
            dependencies.append({"type": "DATABRICKS_VECTOR_INDEX", "name": index_name})
        for key in (
            _DATABRICKS_EMBEDDINGS_ENDPOINT_NAME_KEY,
            _DATABRICKS_LLM_ENDPOINT_NAME_KEY,
            _DATABRICKS_CHAT_ENDPOINT_NAME_KEY,
        ):
            endpoint_names = _fetch_langchain_dependency_from_model_info(
                databricks_dependencies, key
            )
            for endpoint_name in endpoint_names:
                dependencies.append({"type": "DATABRICKS_MODEL_ENDPOINT", "name": endpoint_name})
    return dependencies


def _fetch_langchain_dependency_from_model_resources(databricks_dependencies, key, resource_type):
    dependencies = databricks_dependencies.get(key, [])
    deps = []
    for dependency in dependencies:
        if dependency.get("on_behalf_of_user", False):
            continue
        deps.append({"type": resource_type, "name": dependency["name"]})
    return deps


def _fetch_langchain_dependency_from_model_info(databricks_dependencies, key):
    return databricks_dependencies.get(key, [])


class UcModelRegistryStore(BaseRestStore):
    """
    Client for a remote model registry server accessed via REST API calls

    Args:
        store_uri: URI with scheme 'databricks-uc'
        tracking_uri: URI of the Databricks MLflow tracking server from which to fetch
            run info and download run artifacts, when creating new model
            versions from source artifacts logged to an MLflow run.
    """

    def __init__(self, store_uri, tracking_uri):
        super().__init__(get_host_creds=functools.partial(get_databricks_host_creds, store_uri))
        self.tracking_uri = tracking_uri
        self.get_tracking_host_creds = functools.partial(get_databricks_host_creds, tracking_uri)
        try:
            self.spark = _get_active_spark_session()
        except Exception:
            pass

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
            GenerateTemporaryModelVersionCredentialsRequest: (
                GenerateTemporaryModelVersionCredentialsResponse
            ),
            GetRun: GetRun.Response,
            SetRegisteredModelAliasRequest: SetRegisteredModelAliasResponse,
            DeleteRegisteredModelAliasRequest: DeleteRegisteredModelAliasResponse,
            SetRegisteredModelTagRequest: SetRegisteredModelTagResponse,
            DeleteRegisteredModelTagRequest: DeleteRegisteredModelTagResponse,
            SetModelVersionTagRequest: SetModelVersionTagResponse,
            DeleteModelVersionTagRequest: DeleteModelVersionTagResponse,
            GetModelVersionByAliasRequest: GetModelVersionByAliasResponse,
            CreatePromptRequest: ProtoPrompt,
            SearchPromptsRequest: SearchPromptsResponse,
            DeletePromptRequest: google.protobuf.empty_pb2.Empty,
            SetPromptTagRequest: google.protobuf.empty_pb2.Empty,
            DeletePromptTagRequest: google.protobuf.empty_pb2.Empty,
            CreatePromptVersionRequest: ProtoPromptVersion,
            GetPromptVersionRequest: ProtoPromptVersion,
            DeletePromptVersionRequest: google.protobuf.empty_pb2.Empty,
            GetPromptVersionByAliasRequest: ProtoPromptVersion,
            UpdatePromptRequest: ProtoPrompt,
            GetPromptRequest: ProtoPrompt,
            SearchPromptVersionsRequest: SearchPromptVersionsResponse,
            SetPromptAliasRequest: google.protobuf.empty_pb2.Empty,
            DeletePromptAliasRequest: google.protobuf.empty_pb2.Empty,
            SetPromptVersionTagRequest: google.protobuf.empty_pb2.Empty,
            DeletePromptVersionTagRequest: google.protobuf.empty_pb2.Empty,
            UpdatePromptVersionRequest: ProtoPromptVersion,
            LinkPromptVersionsToModelsRequest: google.protobuf.empty_pb2.Empty,
            LinkPromptsToTracesRequest: google.protobuf.empty_pb2.Empty,
            LinkPromptVersionsToRunsRequest: google.protobuf.empty_pb2.Empty,
        }
        return method_to_response[method]()

    def _get_endpoint_from_method(self, method):
        return _METHOD_TO_INFO[method]

    def _get_all_endpoints_from_method(self, method):
        return _METHOD_TO_ALL_INFO[method]

    # CRUD API for RegisteredModel objects

    def create_registered_model(self, name, tags=None, description=None, deployment_job_id=None):
        """
        Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                instances associated with this registered model.
            description: Description of the model.
            deployment_job_id: Optional deployment job id.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created in the backend.

        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(
            CreateRegisteredModelRequest(
                name=full_name,
                description=description,
                tags=uc_registered_model_tag_from_mlflow_tags(tags),
                deployment_job_id=str(deployment_job_id) if deployment_job_id else None,
            )
        )
        try:
            response_proto = self._call_endpoint(CreateRegisteredModelRequest, req_body)
        except RestException as e:

            def reraise_with_legacy_hint(exception, legacy_hint):
                new_message = exception.message.rstrip(".") + f". {legacy_hint}"
                raise MlflowException(
                    message=new_message,
                    error_code=exception.error_code,
                )

            if "specify all three levels" in e.message:
                # The exception is likely due to the user trying to create a registered model
                # in Unity Catalog without specifying a 3-level name (catalog.schema.model).
                # The user may not be intending to use the Unity Catalog Model Registry at all,
                # but rather the legacy Workspace Model Registry. Accordingly, we re-raise with
                # a hint
                legacy_hint = (
                    "If you are trying to use the legacy Workspace Model Registry, instead of the"
                    " recommended Unity Catalog Model Registry, set the Model Registry URI to"
                    " 'databricks' (legacy) instead of 'databricks-uc' (recommended)."
                )
                reraise_with_legacy_hint(exception=e, legacy_hint=legacy_hint)
            elif "METASTORE_DOES_NOT_EXIST" in e.message:
                legacy_hint = (
                    "If you are trying to use the Model Registry in a Databricks workspace that"
                    " does not have Unity Catalog enabled, either enable Unity Catalog in the"
                    " workspace (recommended) or set the Model Registry URI to 'databricks' to"
                    " use the legacy Workspace Model Registry."
                )
                reraise_with_legacy_hint(exception=e, legacy_hint=legacy_hint)
            else:
                raise

        if deployment_job_id:
            _print_databricks_deployment_job_url(
                model_name=full_name,
                job_id=str(deployment_job_id),
            )
        return registered_model_from_uc_proto(response_proto.registered_model)

    def update_registered_model(self, name, description=None, deployment_job_id=None):
        """
        Update description of the registered model.

        Args:
            name: Registered model name.
            description: New description.
            deployment_job_id: Optional deployment job id.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(
            UpdateRegisteredModelRequest(
                name=full_name,
                description=description,
                deployment_job_id=str(deployment_job_id) if deployment_job_id else None,
            )
        )
        response_proto = self._call_endpoint(UpdateRegisteredModelRequest, req_body)
        if deployment_job_id:
            _print_databricks_deployment_job_url(
                model_name=full_name,
                job_id=str(deployment_job_id),
            )
        return registered_model_from_uc_proto(response_proto.registered_model)

    def rename_registered_model(self, name, new_name):
        """
        Rename the registered model.

        Args:
            name: Registered model name.
            new_name: New proposed name.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(UpdateRegisteredModelRequest(name=full_name, new_name=new_name))
        response_proto = self._call_endpoint(UpdateRegisteredModelRequest, req_body)
        return registered_model_from_uc_proto(response_proto.registered_model)

    def delete_registered_model(self, name):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Registered model name.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(DeleteRegisteredModelRequest(name=full_name))
        self._call_endpoint(DeleteRegisteredModelRequest, req_body)

    def search_registered_models(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for registered models in backend that satisfy the filter criteria.

        Args:
            filter_string: Filter query string, defaults to searching all registered models.
            max_results: Maximum number of registered models desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_registered_models`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
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
            registered_model_search_from_uc_proto(registered_model)
            for registered_model in response_proto.registered_models
        ]
        return PagedList(registered_models, response_proto.next_page_token)

    def get_registered_model(self, name):
        """
        Get registered model instance by name.

        Args:
            name: Registered model name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(GetRegisteredModelRequest(name=full_name))
        response_proto = self._call_endpoint(GetRegisteredModelRequest, req_body)
        return registered_model_from_uc_proto(response_proto.registered_model)

    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        Args:
            name: Registered model name.
            stages: List of desired stages. If input list is None, return latest versions for
                each stage.

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        alias_doc_url = "https://mlflow.org/docs/latest/model-registry.html#deploy-and-organize-models-with-aliases-and-tags"
        if stages is None:
            message = (
                "To load the latest version of a model in Unity Catalog, you can "
                "set an alias on the model version and load it by alias. See "
                f"{alias_doc_url} for details."
            )
        else:
            message = (
                f"Detected attempt to load latest model version in stages {stages}. "
                "You may see this error because:\n"
                "1) You're attempting to load a model version by stage. Setting stages "
                "and loading model versions by stage is unsupported in Unity Catalog. Instead, "
                "use aliases for flexible model deployment. See "
                f"{alias_doc_url} for details.\n"
                "2) You're attempting to load a model version by alias. Use "
                "syntax 'models:/your_model_name@your_alias_name'\n"
                "3) You're attempting load a model version by version number. Verify "
                "that the version number is a valid integer"
            )

        _raise_unsupported_method(
            method="get_latest_versions",
            message=message,
        )

    def set_registered_model_tag(self, name, tag):
        """
        Set a tag for the registered model.

        Args:
            name: Registered model name.
            tag: :py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(
            SetRegisteredModelTagRequest(name=full_name, key=tag.key, value=tag.value)
        )
        self._call_endpoint(SetRegisteredModelTagRequest, req_body)

    def delete_registered_model_tag(self, name, key):
        """
        Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(DeleteRegisteredModelTagRequest(name=full_name, key=key))
        self._call_endpoint(DeleteRegisteredModelTagRequest, req_body)

    # CRUD API for ModelVersion objects
    def _finalize_model_version(self, name, version):
        """
        Finalize a UC model version after its files have been written to managed storage,
        updating its status from PENDING_REGISTRATION to READY

        Args:
            name: Registered model name
            version: Model version number

        Returns:
            Protobuf ModelVersion describing the finalized model version
        """
        req_body = message_to_json(FinalizeModelVersionRequest(name=name, version=version))
        return self._call_endpoint(FinalizeModelVersionRequest, req_body).model_version

    def _get_temporary_model_version_write_credentials(self, name, version) -> TemporaryCredentials:
        """
        Get temporary credentials for uploading model version files

        Args:
            name: Registered model name.
            version: Model version number.

        Returns:
            mlflow.protos.databricks_uc_registry_messages_pb2.TemporaryCredentials containing
            temporary model version credentials.
        """
        req_body = message_to_json(
            GenerateTemporaryModelVersionCredentialsRequest(
                name=name, version=version, operation=MODEL_VERSION_OPERATION_READ_WRITE
            )
        )
        return self._call_endpoint(
            GenerateTemporaryModelVersionCredentialsRequest, req_body
        ).credentials

    def _get_run_and_headers(self, run_id):
        if run_id is None or not is_databricks_uri(self.tracking_uri):
            return None, None
        host_creds = self.get_tracking_host_creds()
        endpoint, method = _TRACKING_METHOD_TO_INFO[GetRun]
        response = http_request(
            host_creds=host_creds,
            endpoint=endpoint,
            method=method,
            params={"run_id": run_id},
        )
        try:
            verify_rest_response(response, endpoint)
        except MlflowException:
            _logger.warning(
                f"Unable to fetch model version's source run (with ID {run_id}) "
                "from tracking server. The source run may be deleted or inaccessible to the "
                "current user. No run link will be recorded for the model version."
            )
            return None, None
        headers = response.headers
        js_dict = response.json()
        parsed_response = GetRun.Response()
        parse_dict(js_dict=js_dict, message=parsed_response)
        run = Run.from_proto(parsed_response.run)
        return headers, run

    def _get_workspace_id(self, headers):
        if headers is None or _DATABRICKS_ORG_ID_HEADER not in headers:
            _logger.warning(
                "Unable to get model version source run's workspace ID from request headers. "
                "No run link will be recorded for the model version"
            )
            return None
        return headers[_DATABRICKS_ORG_ID_HEADER]

    def _get_notebook_id(self, run):
        if run is None:
            return None
        return run.data.tags.get(MLFLOW_DATABRICKS_NOTEBOOK_ID, None)

    def _get_job_id(self, run):
        if run is None:
            return None
        return run.data.tags.get(MLFLOW_DATABRICKS_JOB_ID, None)

    def _get_job_run_id(self, run):
        if run is None:
            return None
        return run.data.tags.get(MLFLOW_DATABRICKS_JOB_RUN_ID, None)

    def _get_lineage_input_sources(self, run):
        from mlflow.data.delta_dataset_source import DeltaDatasetSource

        if run is None:
            return None
        securable_list = []
        if run.inputs is not None:
            for dataset in run.inputs.dataset_inputs:
                dataset_source = mlflow.data.get_source(dataset)
                if (
                    isinstance(dataset_source, DeltaDatasetSource)
                    and dataset_source._get_source_type() == _DELTA_TABLE
                ):
                    # check if dataset is a uc table and then append
                    if dataset_source.delta_table_name and dataset_source.delta_table_id:
                        table_entity = Table(
                            name=dataset_source.delta_table_name,
                            table_id=dataset_source.delta_table_id,
                        )
                        securable_list.append(Securable(table=table_entity))
            if len(securable_list) > _MAX_LINEAGE_DATA_SOURCES:
                _logger.warning(
                    f"Model version has {len(securable_list)!s} upstream datasets, which "
                    f"exceeds the max of 10 upstream datasets for lineage tracking. Only "
                    f"the first 10 datasets will be propagated to Unity Catalog lineage"
                )
            return securable_list[0:_MAX_LINEAGE_DATA_SOURCES]
        else:
            return None

    def _validate_model_signature(self, local_model_path):
        # Import Model here instead of in the top level, to avoid circular import; the
        # mlflow.models.model module imports from MLflow tracking, which triggers an import of
        # this file during store registry initialization
        model = _load_model(local_model_path)
        signature_required_explanation = (
            "All models in the Unity Catalog must be logged with a "
            "model signature containing both input and output "
            "type specifications. See "
            "https://mlflow.org/docs/latest/model/signatures.html#how-to-log-models-with-signatures"
            " for details on how to log a model with a signature"
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

    def _download_model_weights_if_not_saved(self, local_model_path):
        """
        Transformers models can be saved without the base model weights by setting
        `save_pretrained=False` when saving or logging the model. Such 'weight-less'
        model cannot be directly deployed to model serving, so here we download the
        weights proactively from the HuggingFace hub and save them to the model directory.
        """
        model = _load_model(local_model_path)
        flavor_conf = model.flavors.get("transformers")

        if not flavor_conf:
            return

        from mlflow.transformers.flavor_config import FlavorKey
        from mlflow.transformers.model_io import _MODEL_BINARY_FILE_NAME

        if (
            FlavorKey.MODEL_BINARY in flavor_conf
            and os.path.exists(os.path.join(local_model_path, _MODEL_BINARY_FILE_NAME))
            and FlavorKey.MODEL_REVISION not in flavor_conf
        ):
            # Model weights are already saved
            return

        _logger.info(
            "You are attempting to register a transformers model that does not have persisted "
            "model weights. Attempting to fetch the weights so that the model can be registered "
            "within Unity Catalog."
        )
        try:
            mlflow.transformers.persist_pretrained_model(local_model_path)
        except Exception as e:
            raise MlflowException(
                "Failed to download the model weights from the HuggingFace hub and cannot register "
                "the model in the Unity Catalog. Please ensure that the model was saved with the "
                "correct reference to the HuggingFace hub repository and that you have access to "
                "fetch model weights from the defined repository.",
                error_code=INTERNAL_ERROR,
            ) from e

    @contextmanager
    def _local_model_dir(self, source, local_model_path):
        if local_model_path is not None:
            yield local_model_path
        else:
            try:
                local_model_dir = mlflow.artifacts.download_artifacts(
                    artifact_uri=source, tracking_uri=self.tracking_uri
                )
            except Exception as e:
                raise MlflowException(
                    f"Unable to download model artifacts from source artifact location "
                    f"'{source}' in order to upload them to Unity Catalog. Please ensure "
                    f"the source artifact location exists and that you can download from "
                    f"it via mlflow.artifacts.download_artifacts()"
                ) from e
            try:
                yield local_model_dir
            finally:
                # Clean up temporary model directory at end of block. We assume a temporary
                # model directory was created if the `source` is not a local path
                # (must be downloaded from remote to a temporary directory) and
                # `local_model_dir` is not a FUSE-mounted path. The check for FUSE-mounted
                # paths is important as mlflow.artifacts.download_artifacts() can return
                # a FUSE mounted path equivalent to the (remote) source path in some cases,
                # e.g. return /dbfs/some/path for source dbfs:/some/path.
                if not os.path.exists(source) and not is_fuse_or_uc_volumes_uri(local_model_dir):
                    shutil.rmtree(local_model_dir)

    def _get_logged_model_from_model_id(self, model_id) -> Optional[LoggedModel]:
        # load the MLflow LoggedModel by model_id and
        if model_id is None:
            return None
        return mlflow.get_logged_model(model_id)

    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        local_model_path=None,
        model_id: Optional[str] = None,
    ):
        """
        Create a new model version from given source and run ID.

        Args:
            name: Registered model name.
            source: URI indicating the location of the model artifacts.
            run_id: Run ID from MLflow tracking server that generated the model.
            tags: A list of :py:class:`mlflow.entities.model_registry.ModelVersionTag`
                instances associated with this model version.
            run_link: Link to the run from an MLflow tracking server that generated this model.
            description: Description of the version.
            local_model_path: Local path to the MLflow model, if it's already accessible on the
                local filesystem. Can be used by AbstractStores that upload model version files
                to the model registry to avoid a redundant download from the source location when
                logging and registering a model via a single
                mlflow.<flavor>.log_model(..., registered_model_name) call.
            model_id: The ID of the model (from an Experiment) that is being promoted to a
                registered model version, if applicable.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
            created in the backend.
        """
        _require_arg_unspecified(arg_name="run_link", arg_value=run_link)
        logged_model = self._get_logged_model_from_model_id(model_id)
        if logged_model:
            run_id = logged_model.source_run_id
        headers, run = self._get_run_and_headers(run_id)
        source_workspace_id = self._get_workspace_id(headers)
        notebook_id = self._get_notebook_id(run)
        lineage_securable_list = self._get_lineage_input_sources(run)
        job_id = self._get_job_id(run)
        job_run_id = self._get_job_run_id(run)
        extra_headers = None
        if notebook_id is not None or job_id is not None:
            entity_list = []
            lineage_list = None
            if notebook_id is not None:
                notebook_entity = Notebook(id=str(notebook_id))
                entity_list.append(Entity(notebook=notebook_entity))
            if job_id is not None:
                job_entity = Job(id=job_id, job_run_id=job_run_id)
                entity_list.append(Entity(job=job_entity))
            if lineage_securable_list is not None:
                lineage_list = [Lineage(source_securables=lineage_securable_list)]
            lineage_header_info = LineageHeaderInfo(entities=entity_list, lineages=lineage_list)
            # Base64-encode the header value to ensure it's valid ASCII,
            # similar to JWT (see https://stackoverflow.com/a/40347926)
            header_json = message_to_json(lineage_header_info)
            header_base64 = base64.b64encode(header_json.encode())
            extra_headers = {_DATABRICKS_LINEAGE_ID_HEADER: header_base64}
        full_name = get_full_name_from_sc(name, self.spark)
        with self._local_model_dir(source, local_model_path) as local_model_dir:
            self._validate_model_signature(local_model_dir)
            self._download_model_weights_if_not_saved(local_model_dir)
            feature_deps = get_feature_dependencies(local_model_dir)
            other_model_deps = get_model_version_dependencies(local_model_dir)
            req_body = message_to_json(
                CreateModelVersionRequest(
                    name=full_name,
                    source=source,
                    run_id=run_id,
                    description=description,
                    tags=uc_model_version_tag_from_mlflow_tags(tags),
                    run_tracking_server_id=source_workspace_id,
                    feature_deps=feature_deps,
                    model_version_dependencies=other_model_deps,
                    model_id=model_id,
                )
            )
            model_version = self._call_endpoint(
                CreateModelVersionRequest, req_body, extra_headers=extra_headers
            ).model_version

            store = self._get_artifact_repo(model_version, full_name)
            store.log_artifacts(local_dir=local_model_dir, artifact_path="")
            finalized_mv = self._finalize_model_version(
                name=full_name, version=model_version.version
            )
            return model_version_from_uc_proto(finalized_mv)

    def _get_artifact_repo(self, model_version, model_name=None):
        def base_credential_refresh_def():
            return self._get_temporary_model_version_write_credentials(
                name=model_version.name, version=model_version.version
            )

        if is_databricks_sdk_models_artifact_repository_enabled(self.get_host_creds()):
            return DatabricksSDKModelsArtifactRepository(model_name, model_version.version)

        scoped_token = base_credential_refresh_def()
        if scoped_token.storage_mode == StorageMode.DEFAULT_STORAGE:
            return PresignedUrlArtifactRepository(
                self.get_host_creds(), model_version.name, model_version.version
            )

        return get_artifact_repo_from_storage_info(
            storage_location=model_version.storage_location,
            scoped_token=scoped_token,
            base_credential_refresh_def=base_credential_refresh_def,
        )

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        """
        Update model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automatically moved to the "archived" stage. Only
                valid when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be
                raised.
        """
        _raise_unsupported_method(
            method="transition_model_version_stage",
            message="We recommend using aliases instead of stages for more flexible model "
            "deployment management. You can set an alias on a registered model using "
            "`MlflowClient().set_registered_model_alias(name, alias, version)` and load a model "
            "version by alias using the URI 'models:/your_model_name@your_alias', e.g. "
            "`mlflow.pyfunc.load_model('models:/your_model_name@your_alias')`.",
        )

    def update_model_version(self, name, version, description):
        """
        Update metadata associated with a model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.
            description: New model description.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(
            UpdateModelVersionRequest(name=full_name, version=str(version), description=description)
        )
        response_proto = self._call_endpoint(UpdateModelVersionRequest, req_body)
        return model_version_from_uc_proto(response_proto.model_version)

    def delete_model_version(self, name, version):
        """
        Delete model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(DeleteModelVersionRequest(name=full_name, version=str(version)))
        self._call_endpoint(DeleteModelVersionRequest, req_body)

    def get_model_version(self, name, version):
        """
        Get the model version instance by name and version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(GetModelVersionRequest(name=full_name, version=str(version)))
        response_proto = self._call_endpoint(GetModelVersionRequest, req_body)
        return model_version_from_uc_proto(response_proto.model_version)

    def get_model_version_download_uri(self, name, version):
        """
        Get the download location in Model Registry for this model version.
        NOTE: For first version of Model Registry, since the models are not copied over to another
              location, download URI points to input source path.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single URI location that allows reads for downloading.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(
            GetModelVersionDownloadUriRequest(name=full_name, version=str(version))
        )
        response_proto = self._call_endpoint(GetModelVersionDownloadUriRequest, req_body)
        return response_proto.artifact_uri

    def search_model_versions(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for model versions in backend that satisfy the filter criteria.

        Args:
            filter_string: A filter string expression. Currently supports a single filter
                condition either name of model like ``name = 'model_name'`` or
                ``run_id = '...'``.
            max_results: Maximum number of model versions desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_model_versions`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
            objects that satisfy the search expressions. The pagination token for the next
            page can be obtained via the ``token`` attribute of the object.

        """
        _require_arg_unspecified(arg_name="order_by", arg_value=order_by)
        req_body = message_to_json(
            SearchModelVersionsRequest(
                filter=filter_string, page_token=page_token, max_results=max_results
            )
        )
        response_proto = self._call_endpoint(SearchModelVersionsRequest, req_body)
        model_versions = [
            model_version_search_from_uc_proto(mvd) for mvd in response_proto.model_versions
        ]
        return PagedList(model_versions, response_proto.next_page_token)

    def set_model_version_tag(self, name, version, tag):
        """
        Set a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            tag: :py:class:`mlflow.entities.model_registry.ModelVersionTag` instance to log.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(
            SetModelVersionTagRequest(
                name=full_name, version=str(version), key=tag.key, value=tag.value
            )
        )
        self._call_endpoint(SetModelVersionTagRequest, req_body)

    def delete_model_version_tag(self, name, version, key):
        """
        Delete a tag associated with the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(
            DeleteModelVersionTagRequest(name=full_name, version=version, key=key)
        )
        self._call_endpoint(DeleteModelVersionTagRequest, req_body)

    def set_registered_model_alias(self, name, alias, version):
        """
        Set a registered model alias pointing to a model version.

        Args:
            name: Registered model name.
            alias: Name of the alias.
            version: Registered model version number.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(
            SetRegisteredModelAliasRequest(name=full_name, alias=alias, version=str(version))
        )
        self._call_endpoint(SetRegisteredModelAliasRequest, req_body)

    def delete_registered_model_alias(self, name, alias):
        """
        Delete an alias associated with a registered model.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(DeleteRegisteredModelAliasRequest(name=full_name, alias=alias))
        self._call_endpoint(DeleteRegisteredModelAliasRequest, req_body)

    def get_model_version_by_alias(self, name, alias):
        """
        Get the model version instance by name and alias.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(GetModelVersionByAliasRequest(name=full_name, alias=alias))
        response_proto = self._call_endpoint(GetModelVersionByAliasRequest, req_body)
        return model_version_from_uc_proto(response_proto.model_version)

    def _await_model_version_creation(self, mv, await_creation_for):
        """
        Does not wait for the model version to become READY as a successful creation will
        immediately place the model version in a READY state.
        """

    # Prompt-related method overrides for UC

    def create_prompt(
        self,
        name: str,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> Prompt:
        """
        Create a new prompt in Unity Catalog (metadata only, no initial version).
        """
        # Create a Prompt object with the provided fields
        prompt_proto = ProtoPrompt()
        prompt_proto.name = name
        if description:
            prompt_proto.description = description
        if tags:
            prompt_proto.tags.extend(mlflow_tags_to_proto(tags))

        req_body = message_to_json(
            CreatePromptRequest(
                name=name,
                prompt=prompt_proto,
            )
        )
        response_proto = self._call_endpoint(CreatePromptRequest, req_body)
        return proto_info_to_mlflow_prompt_info(response_proto, tags or {})

    def search_prompts(
        self,
        filter_string: Optional[str] = None,
        max_results: Optional[int] = None,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ) -> PagedList[Prompt]:
        """
        Search for prompts in Unity Catalog.

        Args:
            filter_string: Filter string that must include catalog and schema in the format:
                "catalog = 'catalog_name' AND schema = 'schema_name'"
            max_results: Maximum number of results to return
            order_by: List of fields to order by (not used in current implementation)
            page_token: Token for pagination
        """
        # Parse catalog and schema from filter string
        if filter_string:
            parsed_filter = self._parse_catalog_schema_from_filter(filter_string)
        else:
            raise MlflowException(
                "For Unity Catalog prompt registries, you must specify catalog and schema "
                "in the filter string: \"catalog = 'catalog_name' AND schema = 'schema_name'\"",
                INVALID_PARAMETER_VALUE,
            )

        # Build the request with Unity Catalog schema
        unity_catalog_schema = UnityCatalogSchema(
            catalog_name=parsed_filter.catalog_name, schema_name=parsed_filter.schema_name
        )
        req_body = message_to_json(
            SearchPromptsRequest(
                catalog_schema=unity_catalog_schema,
                filter=parsed_filter.remaining_filter,
                max_results=max_results,
                page_token=page_token,
            )
        )

        response_proto = self._call_endpoint(SearchPromptsRequest, req_body)
        prompts = []
        for prompt_info in response_proto.prompts:
            # For UC, only use the basic prompt info without extra tag fetching
            prompts.append(proto_info_to_mlflow_prompt_info(prompt_info, {}))

        return PagedList(prompts, response_proto.next_page_token)

    def _parse_catalog_schema_from_filter(
        self, filter_string: Optional[str]
    ) -> _CatalogSchemaFilter:
        """
        Parse catalog and schema from filter string for Unity Catalog using regex.

        Expects filter format: "catalog = 'catalog_name' AND schema = 'schema_name'"

        Args:
            filter_string: Filter string containing catalog and schema

        Returns:
            _CatalogSchemaFilter object with catalog_name, schema_name, and remaining_filter

        Raises:
            MlflowException: If filter format is invalid for Unity Catalog
        """
        if not filter_string:
            raise MlflowException(
                "For Unity Catalog prompt registries, you must specify catalog and schema "
                "in the filter string: \"catalog = 'catalog_name' AND schema = 'schema_name'\"",
                INVALID_PARAMETER_VALUE,
            )

        # Use pre-compiled regex patterns for better performance
        catalog_match = _CATALOG_PATTERN.search(filter_string)
        schema_match = _SCHEMA_PATTERN.search(filter_string)

        if not catalog_match or not schema_match:
            raise MlflowException(
                "For Unity Catalog prompt registries, filter string must include both "
                "catalog and schema in the format: "
                "\"catalog = 'catalog_name' AND schema = 'schema_name'\". "
                f"Got: {filter_string}",
                INVALID_PARAMETER_VALUE,
            )

        catalog_name = catalog_match.group(1)
        schema_name = schema_match.group(1)

        # Remove catalog and schema from filter string to get remaining filters
        # First, normalize the filter by splitting on AND and rebuilding
        # without catalog/schema parts
        parts = re.split(r"\s+AND\s+", filter_string, flags=re.IGNORECASE)
        remaining_parts = []

        for part in parts:
            part = part.strip()
            # Skip parts that match catalog or schema patterns
            if not (_CATALOG_PATTERN.match(part) or _SCHEMA_PATTERN.match(part)):
                remaining_parts.append(part)

        # Rejoin the remaining parts
        remaining_filter = " AND ".join(remaining_parts) if remaining_parts else None

        return _CatalogSchemaFilter(catalog_name, schema_name, remaining_filter)

    def delete_prompt(self, name: str) -> None:
        """
        Delete a prompt from Unity Catalog.
        """
        req_body = message_to_json(DeletePromptRequest(name=name))
        endpoint, method = self._get_endpoint_from_method(DeletePromptRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            proto_name=DeletePromptRequest,
        )

    def set_prompt_tag(self, name: str, key: str, value: str) -> None:
        """
        Set a tag on a prompt in Unity Catalog.
        """
        req_body = message_to_json(SetPromptTagRequest(name=name, key=key, value=value))
        endpoint, method = self._get_endpoint_from_method(SetPromptTagRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            key=key,
            proto_name=SetPromptTagRequest,
        )

    def delete_prompt_tag(self, name: str, key: str) -> None:
        """
        Delete a tag from a prompt in Unity Catalog.
        """
        req_body = message_to_json(DeletePromptTagRequest(name=name, key=key))
        endpoint, method = self._get_endpoint_from_method(DeletePromptTagRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            key=key,
            proto_name=DeletePromptTagRequest,
        )

    def get_prompt(self, name: str) -> Optional[Prompt]:
        """
        Get prompt by name from Unity Catalog.
        """
        try:
            req_body = message_to_json(GetPromptRequest(name=name))
            endpoint, method = self._get_endpoint_from_method(GetPromptRequest)
            response_proto = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                name=name,
                proto_name=GetPromptRequest,
            )
            return proto_info_to_mlflow_prompt_info(response_proto, {})
        except Exception as e:
            if isinstance(e, MlflowException) and e.error_code == ErrorCode.Name(
                RESOURCE_DOES_NOT_EXIST
            ):
                return None
            raise

    def create_prompt_version(
        self,
        name: str,
        template: str,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> PromptVersion:
        """
        Create a new prompt version in Unity Catalog.
        """
        # Create a PromptVersion object with the provided fields
        prompt_version_proto = ProtoPromptVersion()
        prompt_version_proto.name = name
        # JSON-encode the template for Unity Catalog server
        prompt_version_proto.template = json.dumps(template)

        # Note: version will be set by the backend when creating a new version
        # We don't set it here as it's generated server-side
        if description:
            prompt_version_proto.description = description
        if tags:
            prompt_version_proto.tags.extend(mlflow_tags_to_proto_version_tags(tags))

        req_body = message_to_json(
            CreatePromptVersionRequest(
                name=name,
                prompt_version=prompt_version_proto,
            )
        )
        endpoint, method = self._get_endpoint_from_method(CreatePromptVersionRequest)
        response_proto = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            proto_name=CreatePromptVersionRequest,
        )
        return proto_to_mlflow_prompt(response_proto)

    def get_prompt_version(self, name: str, version: Union[str, int]) -> Optional[PromptVersion]:
        """
        Get a specific prompt version from Unity Catalog.
        """
        try:
            req_body = message_to_json(GetPromptVersionRequest(name=name, version=str(version)))
            endpoint, method = self._get_endpoint_from_method(GetPromptVersionRequest)
            response_proto = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                name=name,
                version=version,
                proto_name=GetPromptVersionRequest,
            )

            # No longer fetch prompt-level tags - keep them completely separate
            return proto_to_mlflow_prompt(response_proto)
        except Exception as e:
            if isinstance(e, MlflowException) and e.error_code == ErrorCode.Name(
                RESOURCE_DOES_NOT_EXIST
            ):
                return None
            raise

    def delete_prompt_version(self, name: str, version: Union[str, int]) -> None:
        """
        Delete a prompt version from Unity Catalog.
        """
        # Delete the specific version only
        req_body = message_to_json(DeletePromptVersionRequest(name=name, version=str(version)))
        endpoint, method = self._get_endpoint_from_method(DeletePromptVersionRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            version=version,
            proto_name=DeletePromptVersionRequest,
        )

    def search_prompt_versions(
        self, name: str, max_results: Optional[int] = None, page_token: Optional[str] = None
    ) -> SearchPromptVersionsResponse:
        """
        Search prompt versions for a given prompt name in Unity Catalog.

        Note: Unity Catalog server uses a non-standard endpoint pattern for this operation.

        Args:
            name: Name of the prompt to search versions for
            max_results: Maximum number of versions to return
            page_token: Token for pagination

        Returns:
            SearchPromptVersionsResponse containing the list of versions
        """
        req_body = message_to_json(
            SearchPromptVersionsRequest(name=name, max_results=max_results, page_token=page_token)
        )
        endpoint, method = self._get_endpoint_from_method(SearchPromptVersionsRequest)
        return self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            proto_name=SearchPromptVersionsRequest,
        )

    def set_prompt_version_tag(
        self, name: str, version: Union[str, int], key: str, value: str
    ) -> None:
        """
        Set a tag on a prompt version in Unity Catalog.
        """
        req_body = message_to_json(
            SetPromptVersionTagRequest(name=name, version=str(version), key=key, value=value)
        )
        endpoint, method = self._get_endpoint_from_method(SetPromptVersionTagRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            version=version,
            key=key,
            proto_name=SetPromptVersionTagRequest,
        )

    def delete_prompt_version_tag(self, name: str, version: Union[str, int], key: str) -> None:
        """
        Delete a tag from a prompt version in Unity Catalog.
        """
        req_body = message_to_json(
            DeletePromptVersionTagRequest(name=name, version=str(version), key=key)
        )
        endpoint, method = self._get_endpoint_from_method(DeletePromptVersionTagRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            version=version,
            key=key,
            proto_name=DeletePromptVersionTagRequest,
        )

    def get_prompt_version_by_alias(self, name: str, alias: str) -> Optional[PromptVersion]:
        """
        Get a prompt version by alias from Unity Catalog.
        """
        try:
            req_body = message_to_json(GetPromptVersionByAliasRequest(name=name, alias=alias))
            endpoint, method = self._get_endpoint_from_method(GetPromptVersionByAliasRequest)
            response_proto = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                name=name,
                alias=alias,
                proto_name=GetPromptVersionByAliasRequest,
            )

            # No longer fetch prompt-level tags - keep them completely separate
            return proto_to_mlflow_prompt(response_proto)
        except Exception as e:
            if isinstance(e, MlflowException) and e.error_code == ErrorCode.Name(
                RESOURCE_DOES_NOT_EXIST
            ):
                return None
            raise

    def set_prompt_alias(self, name: str, alias: str, version: Union[str, int]) -> None:
        """
        Set an alias for a prompt version in Unity Catalog.
        """
        req_body = message_to_json(
            SetPromptAliasRequest(name=name, alias=alias, version=str(version))
        )
        endpoint, method = self._get_endpoint_from_method(SetPromptAliasRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            alias=alias,
            version=version,
            proto_name=SetPromptAliasRequest,
        )

    def delete_prompt_alias(self, name: str, alias: str) -> None:
        """
        Delete an alias from a prompt in Unity Catalog.
        """
        req_body = message_to_json(DeletePromptAliasRequest(name=name, alias=alias))
        endpoint, method = self._get_endpoint_from_method(DeletePromptAliasRequest)
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            name=name,
            alias=alias,
            proto_name=DeletePromptAliasRequest,
        )

    def link_prompt_version_to_model(self, name: str, version: str, model_id: str) -> None:
        """
        Link a prompt version to a model in Unity Catalog.

        Args:
            name: Name of the prompt.
            version: Version of the prompt to link.
            model_id: ID of the model to link to.
        """
        # Call the default implementation, since the LinkPromptVersionsToModels API
        # will initially be a no-op until the Databricks backend supports it
        super().link_prompt_version_to_model(name=name, version=version, model_id=model_id)

        prompt_version_entry = PromptVersionLinkEntry(name=name, version=version)
        req_body = message_to_json(
            LinkPromptVersionsToModelsRequest(
                prompt_versions=[prompt_version_entry], model_ids=[model_id]
            )
        )
        endpoint, method = self._get_endpoint_from_method(LinkPromptVersionsToModelsRequest)
        try:
            # NB: This will not raise an exception if the backend does not support linking.
            # We do this to prioritize reduction in errors and log spam while the prompt
            # registry remains experimental
            self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                name=name,
                version=version,
                model_id=model_id,
                proto_name=LinkPromptVersionsToModelsRequest,
            )
        except Exception:
            _logger.debug("Failed to link prompt version to model in unity catalog", exc_info=True)

    def link_prompts_to_trace(self, prompt_versions: list[PromptVersion], trace_id: str) -> None:
        """
        Link multiple prompt versions to a trace in Unity Catalog.

        Args:
            prompt_versions: List of PromptVersion objects to link.
            trace_id: Trace ID to link to each prompt version.
        """
        super().link_prompts_to_trace(prompt_versions=prompt_versions, trace_id=trace_id)

        prompt_version_entries = [
            PromptVersionLinkEntry(name=pv.name, version=str(pv.version)) for pv in prompt_versions
        ]

        batch_size = 25
        endpoint, method = self._get_endpoint_from_method(LinkPromptsToTracesRequest)

        for i in range(0, len(prompt_version_entries), batch_size):
            batch = prompt_version_entries[i : i + batch_size]
            req_body = message_to_json(
                LinkPromptsToTracesRequest(prompt_versions=batch, trace_ids=[trace_id])
            )
            try:
                self._edit_endpoint_and_call(
                    endpoint=endpoint,
                    method=method,
                    req_body=req_body,
                    proto_name=LinkPromptsToTracesRequest,
                )
            except Exception:
                _logger.debug("Failed to link prompts to traces in unity catalog", exc_info=True)

    def link_prompt_version_to_run(self, name: str, version: str, run_id: str) -> None:
        """
        Link a prompt version to a run in Unity Catalog.

        Args:
            name: Name of the prompt.
            version: Version of the prompt to link.
            run_id: ID of the run to link to.
        """
        super().link_prompt_version_to_run(name=name, version=version, run_id=run_id)

        prompt_version_entry = PromptVersionLinkEntry(name=name, version=version)
        endpoint, method = self._get_endpoint_from_method(LinkPromptVersionsToRunsRequest)

        req_body = message_to_json(
            LinkPromptVersionsToRunsRequest(
                prompt_versions=[prompt_version_entry], run_ids=[run_id]
            )
        )
        try:
            self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=req_body,
                proto_name=LinkPromptVersionsToRunsRequest,
            )
        except Exception:
            _logger.debug("Failed to link prompt version to run in unity catalog", exc_info=True)

    def _edit_endpoint_and_call(self, endpoint, method, req_body, proto_name, **kwargs):
        """
        Edit endpoint URL with parameters and make the call.

        Args:
            endpoint: URL template with placeholders like {name}, {key}
            method: HTTP method
            req_body: Request body
            proto_name: Protobuf message class for response
            **kwargs: Parameters to substitute in the endpoint template
        """
        # Replace placeholders in endpoint with actual values
        for key, value in kwargs.items():
            if value is not None:
                endpoint = endpoint.replace(f"{{{key}}}", str(value))

        # Make the API call
        return call_endpoint(
            self.get_host_creds(),
            endpoint=endpoint,
            method=method,
            json_body=req_body,
            response_proto=self._get_response_from_method(proto_name),
        )
