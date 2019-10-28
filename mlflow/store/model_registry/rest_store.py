from mlflow.entities.model_registry import RegisteredModel, RegisteredModelDetailed, \
    ModelVersion, ModelVersionDetailed
from mlflow.protos.model_registry_pb2 import ModelRegistryService, CreateRegisteredModel, \
    UpdateRegisteredModel, DeleteRegisteredModel, ListRegisteredModels, \
    GetRegisteredModelDetails, GetLatestVersions, CreateModelVersion, UpdateModelVersion, \
    DeleteModelVersion, GetModelVersionDetails, GetModelVersionDownloadUri, SearchModelVersions, \
    GetModelVersionStages
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import call_endpoint, extract_api_info_for_service

_PATH_PREFIX = "/api/2.0"
_METHOD_TO_INFO = extract_api_info_for_service(ModelRegistryService, _PATH_PREFIX)


class RestStore(AbstractStore):
    """
    Note:: Experimental: This entity may change or be removed in a future release without warning.
    Client for a remote model registry server accessed via REST API calls

    :param get_host_creds: Method to be invoked prior to every REST request to get the
      :py:class:`mlflow.rest_utils.MlflowHostCreds` for the request. Note that this
      is a function so that we can obtain fresh credentials in the case of expiry.
    """

    def __init__(self, get_host_creds):
        super(RestStore, self).__init__()
        self.get_host_creds = get_host_creds

    def _call_endpoint(self, api, json_body):
        endpoint, method = _METHOD_TO_INFO[api]
        response_proto = api.Response()
        return call_endpoint(self.get_host_creds(), endpoint, method, json_body, response_proto)

    # CRUD API for RegisteredModel objects

    def create_registered_model(self, name):
        """
        Create a new registered model in backend store.

        :param name: Name of the new model. This is expected to be unique in the backend store.

        :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
        created in the backend.
        """
        req_body = message_to_json(CreateRegisteredModel(name=name))
        response_proto = self._call_endpoint(CreateRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    def update_registered_model(self, registered_model, new_name=None, description=None):
        """
        Updates metadata for RegisteredModel entity. Either ``new_name`` or ``description`` should
        be non-None. Backend raises exception if a registered model with given name does not exist.

        :param registered_model: :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        :param new_name: (Optional) New proposed name for the registered model.
        :param description: (Optional) New description.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        req_body = message_to_json(UpdateRegisteredModel(
            registered_model=registered_model.to_proto(), name=new_name, description=description))
        response_proto = self._call_endpoint(UpdateRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    def delete_registered_model(self, registered_model):
        """
        Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param registered_model: :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        :return: None
        """
        req_body = message_to_json(DeleteRegisteredModel(
            registered_model=registered_model.to_proto()))
        self._call_endpoint(DeleteRegisteredModel, req_body)

    def list_registered_models(self):
        """
        List of all registered models.

        :return: List of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects.
        """
        req_body = message_to_json(ListRegisteredModels())
        response_proto = self._call_endpoint(ListRegisteredModels, req_body)
        return [RegisteredModelDetailed.from_proto(registered_model_detailed)
                for registered_model_detailed in response_proto.registered_models_detailed]

    def get_registered_model_details(self, registered_model):
        """
        :param registered_model: :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModelDetailed` object.
        """
        req_body = message_to_json(GetRegisteredModelDetails(
            registered_model=registered_model.to_proto()))
        response_proto = self._call_endpoint(GetRegisteredModelDetails, req_body)
        return RegisteredModelDetailed.from_proto(response_proto.registered_model_detailed)

    def get_latest_versions(self, registered_model, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        :param registered_model: :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for 'Staging' and 'Production' stages.

        :return: List of `:py:class:`mlflow.entities.model_registry.ModelVersionDetailed` objects.
        """
        req_body = message_to_json(GetLatestVersions(
            registered_model=registered_model.to_proto(), stages=stages))
        response_proto = self._call_endpoint(GetLatestVersions, req_body)
        return [ModelVersionDetailed.from_proto(model_version_detailed)
                for model_version_detailed in response_proto.model_versions_detailed]

    # CRUD API for ModelVersion objects

    def create_model_version(self, name, source, run_id):
        """
        Create a new model version from given source and run ID.

        :param name: Name ID for containing registered model.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model

        :return: A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
        created in the backend.
        """
        req_body = message_to_json(CreateModelVersion(name=name, source=source, run_id=run_id))
        response_proto = self._call_endpoint(CreateModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def update_model_version(self, model_version, stage=None, description=None):
        """
        Update metadata associated with a model version in backend.

        :param model_version: :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        :param stage: New desired stage for this model version.
        :param description: New description.

        :return: None.
        """
        req_body = message_to_json(UpdateModelVersion(model_version=model_version.to_proto(),
                                                      stage=stage, description=description))
        self._call_endpoint(UpdateModelVersion, req_body)

    def delete_model_version(self, model_version):
        """
        Delete model version in backend.

        :param model_version: :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        :return: None
        """
        req_body = message_to_json(DeleteModelVersion(model_version=model_version.to_proto()))
        self._call_endpoint(DeleteModelVersion, req_body)

    def get_model_version_details(self, model_version):
        """
        :param model_version: :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersionDetailed` object.
        """
        req_body = message_to_json(GetModelVersionDetails(model_version=model_version.to_proto()))
        response_proto = self._call_endpoint(GetModelVersionDetails, req_body)
        return ModelVersionDetailed.from_proto(response_proto.model_version_detailed)

    def get_model_version_download_uri(self, model_version):
        """
        Get the download location in Model Registry for this model version.

        :param model_version: :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        :return: A single URI location that allows reads for downloading.
        """
        req_body = message_to_json(GetModelVersionDownloadUri(
            model_version=model_version.to_proto()))
        response_proto = self._call_endpoint(GetModelVersionDownloadUri, req_body)
        return response_proto.artifact_uri

    def search_model_versions(self, filter_string):
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: A filter string expression. Currently supports a single filter
                              condition either name of model like ``name = 'model_name'`` or
                              ``run_id = '...'``.

        :return: PagedList of :py:class:`mlflow.entities.model_registry.ModelVersionDetailed`
                 objects.
        """
        req_body = message_to_json(SearchModelVersions(filter=filter_string))
        response_proto = self._call_endpoint(SearchModelVersions, req_body)
        model_versions_detailed = [ModelVersionDetailed.from_proto(mvd)
                                   for mvd in response_proto.model_versions_detailed]
        return PagedList(model_versions_detailed, None)

    def get_model_version_stages(self, model_version):
        """
        :return: A list of valid stages.
        """
        req_body = message_to_json(GetModelVersionStages(model_version=model_version.to_proto()))
        response_proto = self._call_endpoint(GetModelVersionStages, req_body)
        return response_proto.stages
