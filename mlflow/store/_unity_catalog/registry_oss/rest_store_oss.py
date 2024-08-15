import functools

from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    UpdateRegisteredModel,
    UpdateModelVersion,
    CreateRegisteredModel,
    CreateModelVersion,
    GetRegisteredModel,
    GetModelVersion,
    DeleteRegisteredModel,
    DeleteModelVersion,
    ModelVersionInfo,
    RegisteredModelInfo,
    FinalizeModelVersion,
    TagKeyValue,
)



from mlflow.protos.unity_catalog_oss_service_pb2 import UnityCatalogService
from mlflow.store.model_registry.base_rest_store import BaseRestStore
from mlflow.utils._unity_catalog_oss_utils import registered_model_from_uc_oss_proto, model_version_from_uc_oss_proto
from mlflow.utils._unity_catalog_utils import get_full_name_from_sc
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds

from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    call_endpoint,
    _UC_OSS_REST_API_PATH_PREFIX,
    extract_all_api_info_for_service,
    extract_api_info_for_service,
)

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
            FinalizeModelVersion: ModelVersionInfo,
            UpdateModelVersion: ModelVersionInfo,
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

    def search_registered_models(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        raise NotImplementedError("Method not implemented")

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
        endpoint, method = _METHOD_TO_INFO[FinalizeModelVersion]
        final_endpoint = endpoint.replace("{full_name_arg}", full_name).replace("{version_arg}", str(model_version.version))
        finalize_req_body = message_to_json(FinalizeModelVersion(full_name_arg=full_name, version_arg=model_version.version))
        registered_model_version = call_endpoint(get_databricks_host_creds(), endpoint=final_endpoint, method=method, json_body=finalize_req_body, response_proto=self._get_response_from_method(FinalizeModelVersion))
        return model_version_from_uc_oss_proto(registered_model_version)

    
    def update_model_version(self, name, version, description):
        full_name = get_full_name_from_sc(name, None)
        [catalog_name, schema_name, model_name] = full_name.split(".")
        req_body = message_to_json(
            ModelVersionInfo(
                comment = description,
            )
        )
        endpoint, method = _METHOD_TO_INFO[UpdateModelVersion]
        final_endpoint = endpoint.replace("{full_name_arg}", full_name).replace("{version_arg}", str(version))
        registered_model_version = call_endpoint(get_databricks_host_creds(), endpoint=final_endpoint, method=method, json_body=req_body, response_proto=self._get_response_from_method(UpdateModelVersion))
        return model_version_from_uc_oss_proto(registered_model_version)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        raise NotImplementedError("Method not implemented")

    def delete_model_version(self, name, version):
        full_name = get_full_name_from_sc(name, None)
        req_body = message_to_json(DeleteModelVersion(full_name_arg=full_name, version_arg=version))
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
        

    def get_model_version_download_uri(self, name, version):
        raise NotImplementedError("Method not implemented")

    def search_model_versions(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        raise NotImplementedError("Method not implemented")

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