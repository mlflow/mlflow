import logging

from mlflow.entities.model_registry import RegisteredModel, ModelVersion
from mlflow.protos.model_registry_pb2 import ModelRegistryService, CreateRegisteredModel, \
    UpdateRegisteredModel, DeleteRegisteredModel, ListRegisteredModels, \
    GetLatestVersions, CreateModelVersion, UpdateModelVersion, \
    DeleteModelVersion, GetModelVersionDownloadUri, SearchModelVersions, \
    RenameRegisteredModel, GetRegisteredModel, GetModelVersion, TransitionModelVersionStage
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.abstract_store import AbstractStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import call_endpoint, extract_api_info_for_service

_PATH_PREFIX = "/api/2.0"
_METHOD_TO_INFO = extract_api_info_for_service(ModelRegistryService, _PATH_PREFIX)


_logger = logging.getLogger(__name__)


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

    def update_registered_model(self, name, description):
        """
        Updates metadata for RegisteredModel entity.

        :param name: :py:string: Registered model name.

        :param description: New model description.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        req_body = message_to_json(UpdateRegisteredModel(
            name=name, description=description))
        response_proto = self._call_endpoint(UpdateRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    def rename_registered_model(self, name, new_name):
        """
        Renames the registered model.

        :param name: Registered model name.

        :param new_name: New proposed name for the registered model.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        req_body = message_to_json(RenameRegisteredModel(
            name=name, new_name=new_name))
        response_proto = self._call_endpoint(RenameRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    def delete_registered_model(self, name):
        """
        Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param registered_model: :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        :return: None
        """
        req_body = message_to_json(DeleteRegisteredModel(
            name=name))
        self._call_endpoint(DeleteRegisteredModel, req_body)

    def list_registered_models(self):
        """
        List of all registered models.

        :return: List of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects.
        """
        req_body = message_to_json(ListRegisteredModels())
        response_proto = self._call_endpoint(ListRegisteredModels, req_body)
        return [RegisteredModel.from_proto(registered_model)
                for registered_model in response_proto.registered_models]

    def get_registered_model(self, name):
        """
        :param name: Registered model name.

        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        req_body = message_to_json(GetRegisteredModel(name=name))
        response_proto = self._call_endpoint(GetRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        :param name: Registered model name.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for 'Staging' and 'Production' stages.

        :return: List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        req_body = message_to_json(GetLatestVersions(name=name, stages=stages))
        response_proto = self._call_endpoint(GetLatestVersions, req_body)
        return [ModelVersion.from_proto(model_version)
                for model_version in response_proto.model_versions]

    # CRUD API for ModelVersion objects

    def create_model_version(self, name, source, run_id):
        """
        Create a new model version from given source and run ID.

        :param name: Registered model name.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model

        :return: A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
        created in the backend.
        """
        req_body = message_to_json(CreateModelVersion(name=name, source=source, run_id=run_id))
        response_proto = self._call_endpoint(CreateModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def transition_model_version_stage(self, name, version, stage,
                                       archive_existing_versions):
        """
        Update model version stage.

        :param name: Registered model name.
        :param version: Registered model version.
        :param new_stage: New desired stage for this model version.
        :param archive_existing_versions: If this flag is set, all existing model
        versions in the stage will be atomically moved to the "archived" stage.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        req_body = message_to_json(TransitionModelVersionStage(
            name=name, version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing_versions))
        response_proto = self._call_endpoint(TransitionModelVersionStage, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def update_model_version(self, name, version, description):
        """
        Update metadata associated with a model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.
        :param description: New description.

        :return: None.
        """
        req_body = message_to_json(UpdateModelVersion(name=name, version=str(version),
                                                      description=description))
        response_proto = self._call_endpoint(UpdateModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def delete_model_version(self, name, version):
        """
        Delete model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.

        :return: None
        """
        req_body = message_to_json(DeleteModelVersion(name=name, version=str(version)))
        self._call_endpoint(DeleteModelVersion, req_body)

    def get_model_version(self, name, version):
        """
        :param name: Registered model name.
        :param version: Registered model version.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        req_body = message_to_json(GetModelVersion(name=name, version=str(version)))
        response_proto = self._call_endpoint(GetModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def get_model_version_download_uri(self, name, version):
        """
        Get the download location in Model Registry for this model version.

        :param name: Registered model name.
        :param version: Registered model version.

        :return: A single URI location that allows reads for downloading.
        """
        req_body = message_to_json(GetModelVersionDownloadUri(name=name, version=str(version)))
        response_proto = self._call_endpoint(GetModelVersionDownloadUri, req_body)
        return response_proto.artifact_uri

    def search_model_versions(self, filter_string):
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: A filter string expression. Currently supports a single filter
                              condition either name of model like ``name = 'model_name'`` or
                              ``run_id = '...'``.

        :return: PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 objects.
        """
        req_body = message_to_json(SearchModelVersions(filter=filter_string))
        response_proto = self._call_endpoint(SearchModelVersions, req_body)
        model_versions = [ModelVersion.from_proto(mvd)
                          for mvd in response_proto.model_versions]
        return PagedList(model_versions, None)
