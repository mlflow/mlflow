import functools

from mlflow.protos.unity_catalog_oss_messages_pb2 import (
    CreateRegisteredModel,
    RegisteredModelInfo,
    TagKeyValue,
)
from mlflow.protos.unity_catalog_oss_service_pb2 import UnityCatalogService
from mlflow.store.model_registry.base_rest_store import BaseRestStore
from mlflow.utils._unity_catalog_oss_utils import registered_model_from_uc_oss_proto
from mlflow.utils._unity_catalog_utils import get_full_name_from_sc
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
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
        raise NotImplementedError("Method not implemented")

    def rename_registered_model(self, name, new_name):
        raise NotImplementedError("Method not implemented")

    def delete_registered_model(self, name):
        raise NotImplementedError("Method not implemented")

    def search_registered_models(
        self, filter_string=None, max_results=None, order_by=None, page_token=None
    ):
        raise NotImplementedError("Method not implemented")

    def get_registered_model(self, name):
        raise NotImplementedError("Method not implemented")

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
        raise NotImplementedError("Method not implemented")

    def update_model_version(self, name, version, description):
        raise NotImplementedError("Method not implemented")

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        raise NotImplementedError("Method not implemented")

    def delete_model_version(self, name, version):
        raise NotImplementedError("Method not implemented")

    def get_model_version(self, name, version):
        raise NotImplementedError("Method not implemented")

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
