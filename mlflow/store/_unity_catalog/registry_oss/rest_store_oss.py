import functools
from contextlib import contextmanager
import base64
import os
import shutil
from mlflow.entities import Run
from mlflow.exceptions import MlflowException

from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    UpdateRegisteredModel,
    CreateRegisteredModel,
    CreateModelVersion,
    GetRegisteredModel,
    GetModelVersion,
    DeleteRegisteredModel,
    DeleteModelVersion,
    ModelVersionInfo,
    RegisteredModelInfo,
    TagKeyValue,
)


from mlflow.protos.service_pb2 import GetRun, MlflowService
from mlflow.store._unity_catalog.lineage.constants import (
    _DATABRICKS_LINEAGE_ID_HEADER,
    _DATABRICKS_ORG_ID_HEADER,
)

from mlflow.protos.unity_catalog_oss_service_pb2 import UnityCatalogService
from mlflow.store.model_registry.base_rest_store import BaseRestStore
from mlflow.utils._unity_catalog_oss_utils import registered_model_from_uc_oss_proto, model_version_from_uc_oss_proto
from mlflow.utils._unity_catalog_utils import get_full_name_from_sc
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds, is_databricks_uri
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATABRICKS_JOB_ID,
    MLFLOW_DATABRICKS_JOB_RUN_ID,
    MLFLOW_DATABRICKS_NOTEBOOK_ID,
)
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    call_endpoint,
    _UC_OSS_REST_API_PATH_PREFIX,
    _REST_API_PATH_PREFIX,
    extract_all_api_info_for_service,
    extract_api_info_for_service,
)

_TRACKING_METHOD_TO_INFO = extract_api_info_for_service(MlflowService, _REST_API_PATH_PREFIX)
_METHOD_TO_INFO = extract_api_info_for_service(UnityCatalogService, _UC_OSS_REST_API_PATH_PREFIX)
_METHOD_TO_ALL_INFO = extract_all_api_info_for_service(
    UnityCatalogService, _UC_OSS_REST_API_PATH_PREFIX
)


@experimental
class UnityCatalogOssStore(BaseRestStore):
    """
    Client for an Open Source Unity Catalog Server accessed via REST API calls.
    """

    def __init__(self, store_uri):
        super().__init__(get_host_creds=functools.partial(get_databricks_host_creds, store_uri))

    def _get_response_from_method(self, method):
        method_to_response = {
            CreateRegisteredModel: RegisteredModelInfo,
            CreateModelVersion: ModelVersionInfo,
            UpdateRegisteredModel: RegisteredModelInfo,
            DeleteRegisteredModel: DeleteRegisteredModel,  # DeleteRegisteredModel does not return a response
            DeleteModelVersion: DeleteModelVersion,  # DeleteModelVersion does not return a response
            GetRegisteredModel: RegisteredModelInfo,
            GetModelVersion: ModelVersionInfo,
        }
        return method_to_response[method]()

    def _get_endpoint_from_method(self, method):
        return _METHOD_TO_INFO[method]

    def _get_all_endpoints_from_method(self, method):
        return _METHOD_TO_ALL_INFO[method]

    def create_registered_model(self, name, tags=None, description=None):
        full_name = get_full_name_from_sc(name, None)
        [catalog_name, schema_name, model_name] = full_name.split(".")
        comment = description if description else ""
        tags = [TagKeyValue(key=tag.key, value=tag.value) for tag in (tags or [])]
        # RegisteredModelInfo is inlined in the request and the response.
        # https://docs.databricks.com/api/workspace/registeredmodels/create
        req_body = message_to_json(
            RegisteredModelInfo(
                name=model_name,
                catalog_name=catalog_name,
                schema_name=schema_name,
                comment=comment,
                tags=tags,
            )
        )
        registered_model_info = self._call_endpoint(CreateRegisteredModel, req_body)
        return registered_model_from_uc_oss_proto(registered_model_info)

    def update_registered_model(self, name, description):
        full_name = get_full_name_from_sc(name, None)
        [catalog_name, schema_name, model_name] = full_name.split(".")
        comment = description if description else ""
        req_body = message_to_json(
            UpdateRegisteredModel(
            full_name_arg=full_name,
            new_name=model_name,
            registered_model_info=
                RegisteredModelInfo(
                    name=model_name,
                    catalog_name=catalog_name,
                    schema_name=schema_name,
                    comment=comment,
                ),
            )
        )
        endpoint, method = _METHOD_TO_INFO[UpdateRegisteredModel]
        final_endpoint = endpoint.replace("{full_name_arg}", full_name)
        registered_model_info = call_endpoint(get_databricks_host_creds(), endpoint=final_endpoint, method=method, json_body=req_body, response_proto=self._get_response_from_method(UpdateRegisteredModel))
        return registered_model_from_uc_oss_proto(registered_model_info)

    # def rename_registered_model(self, name, new_name):
    #     raise NotImplementedError("Method not implemented")

    def delete_registered_model(self, name):
        full_name = get_full_name_from_sc(name, None)
        req_body = message_to_json(
            DeleteRegisteredModel(
                full_name_arg=full_name,
            ))
        endpoint, method = _METHOD_TO_INFO[DeleteRegisteredModel]
        final_endpoint = endpoint.replace("{full_name_arg}", full_name)
        registered_model_info = call_endpoint(get_databricks_host_creds(), endpoint=final_endpoint, method=method, json_body=req_body, response_proto=self._get_response_from_method(DeleteRegisteredModel))        

    # def search_registered_models(
    #     self, filter_string=None, max_results=None, order_by=None, page_token=None
    # ):
    #     raise NotImplementedError("Method not implemented")

    def get_registered_model(self, name):
        full_name = get_full_name_from_sc(name, None)
        req_body = message_to_json(
            GetRegisteredModel(
                full_name_arg=full_name
            ))
        endpoint, method = _METHOD_TO_INFO[GetRegisteredModel]
        final_endpoint = endpoint.replace("{full_name_arg}", full_name)
        registered_model_info = call_endpoint(get_databricks_host_creds(), endpoint=final_endpoint, method=method, json_body=req_body, response_proto=self._get_response_from_method(GetRegisteredModel))
        return registered_model_from_uc_oss_proto(registered_model_info)

    # def get_latest_versions(self, name, stages=None):
    #     raise NotImplementedError("Method not implemented")

    # def set_registered_model_tag(self, name, tag):
    #     raise NotImplementedError("Method not implemented")

    # def delete_registered_model_tag(self, name, key):
    #     raise NotImplementedError("Method not implemented")

    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        local_model_path=None,
    ):
        full_name = get_full_name_from_sc(name, None)
        [catalog_name, schema_name, model_name] = full_name.split(".")
        req_body = message_to_json(
            ModelVersionInfo(
                model_name = model_name,
                catalog_name = catalog_name,
                schema_name = schema_name,
                source = source,
                run_id = run_id,
                comment = description,
            )
        )
        model_version = self._call_endpoint(CreateModelVersion, req_body)
        return model_version_from_uc_oss_proto(model_version)
    
    def update_model_version(self, name, version, description):
        raise NotImplementedError("Method not implemented")

    # def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
    #     raise NotImplementedError("Method not implemented")

    def delete_model_version(self, name, version):
        full_name = get_full_name_from_sc(name, None)
        req_body = message_to_json(DeleteRegisteredModel(full_name_arg=full_name, version_arg=version))
        endpoint, method = _METHOD_TO_INFO[DeleteModelVersion]
        final_endpoint = endpoint.replace("{full_name_arg}", full_name).replace("{version_arg}", str(version))
        registered_model_info = call_endpoint(get_databricks_host_creds(), endpoint=final_endpoint, method=method, json_body=req_body, response_proto=self._get_response_from_method(DeleteModelVersion))

    def get_model_version(self, name, version):
        full_name = get_full_name_from_sc(name, None)
        req_body = message_to_json(GetModelVersion(full_name_arg=full_name, version_arg=version))
        endpoint, method = _METHOD_TO_INFO[GetModelVersion]
        final_endpoint = endpoint.replace("{full_name_arg}", full_name).replace("{version_arg}", str(version))
        registered_model_version = call_endpoint(get_databricks_host_creds(), endpoint=final_endpoint, method=method, json_body=req_body, response_proto=self._get_response_from_method(GetModelVersion))
        return model_version_from_uc_oss_proto(registered_model_version)
        

    # def get_model_version_download_uri(self, name, version):
    #     raise NotImplementedError("Method not implemented")

    # def search_model_versions(
    #     self, filter_string=None, max_results=None, order_by=None, page_token=None
    # ):
    #     raise NotImplementedError("Method not implemented")

    # def set_model_version_tag(self, name, version, tag):
    #     raise NotImplementedError("Method not implemented")

    # def delete_model_version_tag(self, name, version, key):
    #     raise NotImplementedError("Method not implemented")

    # def set_registered_model_alias(self, name, alias, version):
    #     raise NotImplementedError("Method not implemented")

    # def delete_registered_model_alias(self, name, alias):
    #     raise NotImplementedError("Method not implemented")

    # def get_model_version_by_alias(self, name, alias):
    #     raise NotImplementedError("Method not implemented")

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
