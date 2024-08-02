from mlflow.store.model_registry.base_rest_store import BaseRestStore
from mlflow.utils.annotations import experimental


@experimental
class OssUnityCatalogStore(BaseRestStore):
    """
    Client for an Open Source Unity Catalog Server accessed via REST API calls.
    """

    def _get_all_endpoints_from_method(self, method):
        raise NotImplementedError("Method not implemented")

    def _get_endpoint_from_method(self, method):
        raise NotImplementedError("Method not implemented")

    def _get_response_from_method(self, method):
        raise NotImplementedError("Method not implemented")

    def create_registered_model(self, name, tags=None, description=None):
        raise NotImplementedError("Method not implemented")

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
