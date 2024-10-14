import base64
import functools
import logging
import os
import shutil
from contextlib import contextmanager

import mlflow
from mlflow.entities import Run
from mlflow.environment_variables import MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR
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
from mlflow.store._unity_catalog.lineage.constants import (
    _DATABRICKS_LINEAGE_ID_HEADER,
    _DATABRICKS_ORG_ID_HEADER,
)
from mlflow.store.artifact.databricks_sdk_models_artifact_repo import (
    DatabricksSDKModelsArtifactRepository,
)
from mlflow.store.artifact.presigned_url_artifact_repo import PresignedUrlArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.rest_store import BaseRestStore
from mlflow.utils._spark_utils import _get_active_spark_session
from mlflow.utils._unity_catalog_utils import (
    get_artifact_repo_from_storage_info,
    get_full_name_from_sc,
    model_version_from_uc_proto,
    model_version_search_from_uc_proto,
    registered_model_from_uc_proto,
    registered_model_search_from_uc_proto,
    uc_model_version_tag_from_mlflow_tags,
    uc_registered_model_tag_from_mlflow_tags,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds, is_databricks_uri
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_JOB_RUN_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
)
from mlflow.utils.proto_json_utils import message_to_json, parse_dict
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    extract_all_api_info_for_service,
    extract_api_info_for_service,
    http_request,
    verify_rest_response,
)
from mlflow.utils.uri import is_fuse_or_uc_volumes_uri

_TRACKING_METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _REST_API_PATH_PREFIX)
_METHOD_TO_INFO = extract_api_info_for_service(UcModelRegistryService, _REST_API_PATH_PREFIX)
_METHOD_TO_ALL_INFO = extract_all_api_info_for_service(
    UcModelRegistryService, _REST_API_PATH_PREFIX
)

_logger = logging.getLogger(__name__)
_DELTA_TABLE = "delta_table"
_MAX_LINEAGE_DATA_SOURCES = 10


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

    databricks_resources = getattr(model, "resources", {})

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
                databricks_dependencies, ResourceType.FUNCTION.value, "DATABRICKS_UC_FUNCTION"
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
    for depndency in dependencies:
        deps.append({"type": resource_type, **depndency})
    return deps


def _fetch_langchain_dependency_from_model_info(databricks_dependencies, key):
    return databricks_dependencies.get(key, [])


@experimental
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

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                instances associated with this registered model.
            description: Description of the model.

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
            )
        )
        response_proto = self._call_endpoint(CreateRegisteredModelRequest, req_body)
        return registered_model_from_uc_proto(response_proto.registered_model)

    def update_registered_model(self, name, description):
        """
        Update description of the registered model.

        Args:
            name: Registered model name.
            description: New description.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        full_name = get_full_name_from_sc(name, self.spark)
        req_body = message_to_json(
            UpdateRegisteredModelRequest(name=full_name, description=description)
        )
        response_proto = self._call_endpoint(UpdateRegisteredModelRequest, req_body)
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
            host_creds=host_creds, endpoint=endpoint, method=method, params={"run_id": run_id}
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

    def create_model_version(  # noqa: D417
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        local_model_path=None,
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

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
            created in the backend.
        """
        _require_arg_unspecified(arg_name="run_link", arg_value=run_link)
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

        if MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC.get():
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
            SetModelVersionTagRequest(name=full_name, version=version, key=tag.key, value=tag.value)
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
