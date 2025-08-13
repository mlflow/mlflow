import functools
import os
import shutil
from contextlib import contextmanager

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    READ_WRITE_MODEL_VERSION,
    CreateModelVersion,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteRegisteredModel,
    FinalizeModelVersion,
    GenerateTemporaryModelVersionCredential,
    GetModelVersion,
    GetRegisteredModel,
    ListModelVersions,
    ListRegisteredModels,
    ModelVersionInfo,
    RegisteredModelInfo,
    TemporaryCredentials,
    UpdateModelVersion,
    UpdateRegisteredModel,
)
from mlflow.protos.unity_catalog_oss_service_pb2 import UnityCatalogService
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.base_rest_store import BaseRestStore
from mlflow.utils._unity_catalog_oss_utils import (
    get_model_version_from_uc_oss_proto,
    get_model_version_search_from_uc_oss_proto,
    get_registered_model_from_uc_oss_proto,
    get_registered_model_search_from_uc_oss_proto,
    parse_model_name,
)
from mlflow.utils._unity_catalog_utils import (
    get_artifact_repo_from_storage_info,
    get_full_name_from_sc,
)
from mlflow.utils.oss_registry_utils import get_oss_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _UC_OSS_REST_API_PATH_PREFIX,
    call_endpoint,
    extract_all_api_info_for_service,
    extract_api_info_for_service,
)
from mlflow.utils.uri import is_file_uri, is_fuse_or_uc_volumes_uri

_METHOD_TO_INFO = extract_api_info_for_service(UnityCatalogService, _UC_OSS_REST_API_PATH_PREFIX)
_METHOD_TO_ALL_INFO = extract_all_api_info_for_service(
    UnityCatalogService, _UC_OSS_REST_API_PATH_PREFIX
)


def _raise_unsupported_arg(arg_name, message=None):
    messages = [
        f"Argument '{arg_name}' is unsupported for models in the Unity Catalog.",
    ]
    if message is not None:
        messages.append(message)
    raise MlflowException(" ".join(messages))


def _require_arg_unspecified(arg_name, arg_value, default_values=None, message=None):
    default_values = [None] if default_values is None else default_values
    if arg_value not in default_values:
        _raise_unsupported_arg(arg_name, message)


class UnityCatalogOssStore(BaseRestStore):
    """
    Client for an Open Source Unity Catalog Server accessed via REST API calls.
    """

    def __init__(self, store_uri):
        super().__init__(get_host_creds=functools.partial(get_oss_host_creds, store_uri))
        self.tracking_uri = None  # OSS has no tracking URI

    def _get_response_from_method(self, method):
        method_to_response = {
            CreateRegisteredModel: RegisteredModelInfo,
            CreateModelVersion: ModelVersionInfo,
            UpdateRegisteredModel: RegisteredModelInfo,
            DeleteRegisteredModel: DeleteRegisteredModel,
            DeleteModelVersion: DeleteModelVersion.Response,
            GetRegisteredModel: RegisteredModelInfo,
            GetModelVersion: ModelVersionInfo,
            FinalizeModelVersion: ModelVersionInfo,
            UpdateModelVersion: ModelVersionInfo,
            GenerateTemporaryModelVersionCredential: TemporaryCredentials,
            ListRegisteredModels: ListRegisteredModels.Response,
            ListModelVersions: ListModelVersions.Response,
        }
        return method_to_response[method]()

    def _get_endpoint_from_method(self, method):
        return _METHOD_TO_INFO[method]

    def _get_all_endpoints_from_method(self, method):
        return _METHOD_TO_ALL_INFO[method]

    def create_registered_model(self, name, tags=None, description=None, deployment_job_id=None):
        """
        Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: Not supported for Unity Catalog OSS yet.
            description: Description of the model.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created in the backend.

        """
        [catalog_name, schema_name, model_name] = name.split(".")
        comment = description if description else ""
        # RegisteredModelInfo is inlined in the request and the response.
        # https://docs.databricks.com/api/workspace/registeredmodels/create
        # TODO: Update the above reference to UC OSS documentation when it's available
        req_body = message_to_json(
            CreateRegisteredModel(
                name=model_name,
                catalog_name=catalog_name,
                schema_name=schema_name,
                comment=comment,
            )
        )
        registered_model_info = self._call_endpoint(CreateRegisteredModel, req_body)
        return get_registered_model_from_uc_oss_proto(registered_model_info)

    def update_registered_model(self, name, description, deployment_job_id=None):
        """
        Update description of the registered model.

        Args:
            name: Registered model name.
            description: New description.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        full_name = get_full_name_from_sc(name, None)
        comment = description if description else ""
        req_body = message_to_json(
            UpdateRegisteredModel(
                full_name=full_name,
                comment=comment,
            )
        )
        endpoint, method = _METHOD_TO_INFO[UpdateRegisteredModel]
        registered_model_info = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            full_name=full_name,
            proto_name=UpdateRegisteredModel,
        )
        return get_registered_model_from_uc_oss_proto(registered_model_info)

    def rename_registered_model(self, name, new_name):
        raise NotImplementedError("Method not implemented")

    def delete_registered_model(self, name):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Registered model name.

        Returns:
            None
        """
        full_name = get_full_name_from_sc(name, None)
        req_body = message_to_json(
            DeleteRegisteredModel(
                full_name=full_name,
            )
        )
        endpoint, method = _METHOD_TO_INFO[DeleteRegisteredModel]
        self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            full_name=full_name,
            proto_name=DeleteRegisteredModel,
        )

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
            ListRegisteredModels(
                max_results=max_results,
                page_token=page_token,
            )
        )
        endpoint, method = _METHOD_TO_INFO[ListRegisteredModels]
        response_proto = call_endpoint(
            self.get_host_creds(),
            endpoint=endpoint,
            method=method,
            json_body=req_body,
            response_proto=self._get_response_from_method(ListRegisteredModels),
        )
        registered_models = [
            get_registered_model_search_from_uc_oss_proto(registered_model)
            for registered_model in response_proto.registered_models
        ]
        return PagedList(registered_models, response_proto.next_page_token)

    def get_registered_model(self, name):
        full_name = get_full_name_from_sc(name, None)
        req_body = message_to_json(GetRegisteredModel(full_name=full_name))
        endpoint, method = _METHOD_TO_INFO[GetRegisteredModel]
        registered_model_info = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            full_name=full_name,
            proto_name=GetRegisteredModel,
            version=None,
        )
        return get_registered_model_from_uc_oss_proto(registered_model_info)

    def get_latest_versions(self, name, stages=None):
        raise NotImplementedError("Method not implemented")

    def set_registered_model_tag(self, name, tag):
        raise NotImplementedError("Method not implemented")

    def delete_registered_model_tag(self, name, key):
        raise NotImplementedError("Method not implemented")

    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        local_model_path=None,
        model_id: str | None = None,
    ):
        with self._local_model_dir(source, local_model_path) as local_model_dir:
            [catalog_name, schema_name, model_name] = name.split(".")
            req_body = message_to_json(
                CreateModelVersion(
                    model_name=model_name,
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    source=source,
                    run_id=run_id,
                    comment=description,
                )
            )
            model_version = self._call_endpoint(CreateModelVersion, req_body)
            store = self._get_artifact_repo(model_version)
            store.log_artifacts(local_dir=local_model_dir, artifact_path="")
            endpoint, method = _METHOD_TO_INFO[FinalizeModelVersion]
            finalize_req_body = message_to_json(
                FinalizeModelVersion(full_name=name, version=model_version.version)
            )
            registered_model_version = self._edit_endpoint_and_call(
                endpoint=endpoint,
                method=method,
                req_body=finalize_req_body,
                full_name=name,
                proto_name=FinalizeModelVersion,
                version=model_version.version,
            )
            return get_model_version_from_uc_oss_proto(registered_model_version)

    def update_model_version(self, name, version, description):
        full_name = get_full_name_from_sc(name, None)
        version = int(version)
        req_body = message_to_json(
            UpdateModelVersion(
                full_name=full_name,
                version=version,
                comment=description,
            )
        )
        endpoint, method = _METHOD_TO_INFO[UpdateModelVersion]
        registered_model_version = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            full_name=full_name,
            proto_name=UpdateModelVersion,
            version=version,
        )
        return get_model_version_from_uc_oss_proto(registered_model_version)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        raise NotImplementedError("Method not implemented")

    def delete_model_version(self, name, version):
        full_name = get_full_name_from_sc(name, None)
        version = int(version)
        req_body = message_to_json(DeleteModelVersion(full_name=full_name, version=version))
        endpoint, method = _METHOD_TO_INFO[DeleteModelVersion]
        return self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            full_name=full_name,
            proto_name=FinalizeModelVersion,
            version=version,
        )

    # This method exists to return the actual UC response object,
    # which contains the storage location
    def _get_model_version_endpoint_response(self, name, version):
        full_name = get_full_name_from_sc(name, None)
        version = int(version)
        req_body = message_to_json(GetModelVersion(full_name=full_name, version=version))
        endpoint, method = _METHOD_TO_INFO[GetModelVersion]
        return self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            full_name=full_name,
            proto_name=GetModelVersion,
            version=version,
        )

    def get_model_version(self, name, version):
        return get_model_version_from_uc_oss_proto(
            self._get_model_version_endpoint_response(name, version)
        )

    def search_model_versions(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        """
        Search for model versions in backend that satisfy the filter criteria.

        Args:
            filter_string: A filter string expression. Currently supports a single filter
                condition either name of model like ``name = 'model_name'``
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
        full_name = parse_model_name(filter_string)
        req_body = message_to_json(
            ListModelVersions(full_name=full_name, page_token=page_token, max_results=max_results)
        )
        endpoint, method = _METHOD_TO_INFO[ListModelVersions]
        response_proto = self._edit_endpoint_and_call(
            endpoint=endpoint,
            method=method,
            req_body=req_body,
            full_name=full_name,
            proto_name=ListModelVersions,
        )
        model_versions = [
            get_model_version_search_from_uc_oss_proto(mvd) for mvd in response_proto.model_versions
        ]
        return PagedList(model_versions, response_proto.next_page_token)

    def set_model_version_tag(self, name, version, tag):
        raise NotImplementedError("Method not implemented")

    def delete_model_version_tag(self, name, version, key):
        raise NotImplementedError("Method not implemented")

    def set_registered_model_alias(self, name, alias, version):
        raise NotImplementedError("Method not implemented")

    def delete_registered_model_alias(self, name, alias):
        raise NotImplementedError("Method not implemented")

    def get_model_version_by_alias(self, name, alias):
        raise NotImplementedError("Method not implemented")

    def _get_artifact_repo(self, model_version):
        if is_file_uri(model_version.storage_location):
            return LocalArtifactRepository(artifact_uri=model_version.storage_location)

        def base_credential_refresh_def():
            return self._get_temporary_model_version_write_credentials_oss(
                model_name=model_version.model_name,
                catalog_name=model_version.catalog_name,
                schema_name=model_version.schema_name,
                version=model_version.version,
            )

        scoped_token = base_credential_refresh_def()

        return get_artifact_repo_from_storage_info(
            storage_location=model_version.storage_location,
            scoped_token=scoped_token,
            base_credential_refresh_def=base_credential_refresh_def,
            is_oss=True,
        )

    def _get_temporary_model_version_write_credentials_oss(
        self, model_name, catalog_name, schema_name, version
    ):
        """
        Get temporary credentials for uploading model version files

        Args:
            name: Registered model name.
            version: Model version number.

        Returns:
            mlflow.protos.unity_catalog_oss_messages_pb2.TemporaryCredentials containing
            temporary model version credentials.
        """
        req_body = message_to_json(
            GenerateTemporaryModelVersionCredential(
                catalog_name=catalog_name,
                schema_name=schema_name,
                model_name=model_name,
                version=int(version),
                operation=READ_WRITE_MODEL_VERSION,
            )
        )
        return self._call_endpoint(GenerateTemporaryModelVersionCredential, req_body)

    def get_model_version_download_uri(self, name, version):
        response = self._get_model_version_endpoint_response(name, int(version))
        return response.storage_location

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
                    "Unable to download model artifacts from source artifact location "
                    f"'{source}' in order to upload them to Unity Catalog. Please ensure "
                    "the source artifact location exists and that you can download from "
                    "it via mlflow.artifacts.download_artifacts()"
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

    def _edit_endpoint_and_call(
        self, endpoint, method, req_body, full_name, proto_name, version=None
    ):
        if version is not None:
            endpoint = endpoint.replace("{full_name}", full_name).replace("{version}", str(version))
        else:
            endpoint = endpoint.replace("{full_name}", full_name)
        return call_endpoint(
            self.get_host_creds(),
            endpoint=endpoint,
            method=method,
            json_body=req_body,
            response_proto=self._get_response_from_method(proto_name),
        )
