import logging

from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.protos.model_registry_pb2 import (
    CreateModelVersion,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteModelVersionTag,
    DeleteRegisteredModel,
    DeleteRegisteredModelAlias,
    DeleteRegisteredModelTag,
    GetLatestVersions,
    GetModelVersion,
    GetModelVersionByAlias,
    GetModelVersionDownloadUri,
    GetRegisteredModel,
    ModelRegistryService,
    RenameRegisteredModel,
    SearchModelVersions,
    SearchRegisteredModels,
    SetModelVersionTag,
    SetRegisteredModelAlias,
    SetRegisteredModelTag,
    TransitionModelVersionStage,
    UpdateModelVersion,
    UpdateRegisteredModel,
)
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.model_registry.base_rest_store import BaseRestStore
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.utils.rest_utils import (
    _REST_API_PATH_PREFIX,
    extract_all_api_info_for_service,
    extract_api_info_for_service,
)

_METHOD_TO_INFO = extract_api_info_for_service(ModelRegistryService, _REST_API_PATH_PREFIX)
_METHOD_TO_ALL_INFO = extract_all_api_info_for_service(ModelRegistryService, _REST_API_PATH_PREFIX)

_logger = logging.getLogger(__name__)


class RestStore(BaseRestStore):
    """
    Client for a remote model registry server accessed via REST API calls

    Args:
        get_host_creds: Method to be invoked prior to every REST request to get the
            :py:class:`mlflow.rest_utils.MlflowHostCreds` for the request. Note that this
            is a function so that we can obtain fresh credentials in the case of expiry.
    """

    def _get_response_from_method(self, method):
        return method.Response()

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
        proto_tags = [tag.to_proto() for tag in tags or []]
        req_body = message_to_json(
            CreateRegisteredModel(name=name, tags=proto_tags, description=description)
        )
        response_proto = self._call_endpoint(CreateRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    def update_registered_model(self, name, description):
        """
        Update description of the registered model.

        Args:
            name: Registered model name.
            description: New description.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        req_body = message_to_json(UpdateRegisteredModel(name=name, description=description))
        response_proto = self._call_endpoint(UpdateRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    def rename_registered_model(self, name, new_name):
        """
        Rename the registered model.

        Args:
            name: Registered model name.
            new_name: New proposed name.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
        req_body = message_to_json(RenameRegisteredModel(name=name, new_name=new_name))
        response_proto = self._call_endpoint(RenameRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    def delete_registered_model(self, name):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Registered model name.

        Returns:
            None
        """
        req_body = message_to_json(DeleteRegisteredModel(name=name))
        self._call_endpoint(DeleteRegisteredModel, req_body)

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
        req_body = message_to_json(
            SearchRegisteredModels(
                filter=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )
        )
        response_proto = self._call_endpoint(SearchRegisteredModels, req_body)
        registered_models = [
            RegisteredModel.from_proto(registered_model)
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
        req_body = message_to_json(GetRegisteredModel(name=name))
        response_proto = self._call_endpoint(GetRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

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
        req_body = message_to_json(GetLatestVersions(name=name, stages=stages))
        response_proto = self._call_endpoint(GetLatestVersions, req_body, call_all_endpoints=True)
        return [
            ModelVersion.from_proto(model_version)
            for model_version in response_proto.model_versions
        ]

    def set_registered_model_tag(self, name, tag):
        """
        Set a tag for the registered model.

        Args:
            name: Registered model name.
            tag: :py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.

        Returns:
            None
        """
        req_body = message_to_json(SetRegisteredModelTag(name=name, key=tag.key, value=tag.value))
        self._call_endpoint(SetRegisteredModelTag, req_body)

    def delete_registered_model_tag(self, name, key):
        """
        Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
        """
        req_body = message_to_json(DeleteRegisteredModelTag(name=name, key=key))
        self._call_endpoint(DeleteRegisteredModelTag, req_body)

    # CRUD API for ModelVersion objects

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
        proto_tags = [tag.to_proto() for tag in tags or []]
        req_body = message_to_json(
            CreateModelVersion(
                name=name,
                source=source,
                run_id=run_id,
                run_link=run_link,
                tags=proto_tags,
                description=description,
            )
        )
        response_proto = self._call_endpoint(CreateModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions):
        """
        Update model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automatically moved to the "archived" stage. Only
                valid when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will
                be raised.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """
        req_body = message_to_json(
            TransitionModelVersionStage(
                name=name,
                version=str(version),
                stage=stage,
                archive_existing_versions=archive_existing_versions,
            )
        )
        response_proto = self._call_endpoint(TransitionModelVersionStage, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

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
        req_body = message_to_json(
            UpdateModelVersion(name=name, version=str(version), description=description)
        )
        response_proto = self._call_endpoint(UpdateModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    def delete_model_version(self, name, version):
        """
        Delete model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            None
        """
        req_body = message_to_json(DeleteModelVersion(name=name, version=str(version)))
        self._call_endpoint(DeleteModelVersion, req_body)

    def get_model_version(self, name, version):
        """
        Get the model version instance by name and version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        req_body = message_to_json(GetModelVersion(name=name, version=str(version)))
        response_proto = self._call_endpoint(GetModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

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
        req_body = message_to_json(GetModelVersionDownloadUri(name=name, version=str(version)))
        response_proto = self._call_endpoint(GetModelVersionDownloadUri, req_body)
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
        req_body = message_to_json(
            SearchModelVersions(
                filter=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )
        )
        response_proto = self._call_endpoint(SearchModelVersions, req_body)
        model_versions = [ModelVersion.from_proto(mvd) for mvd in response_proto.model_versions]
        return PagedList(model_versions, response_proto.next_page_token)

    def set_model_version_tag(self, name, version, tag):
        """
        Set a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            tag: :py:class:`mlflow.entities.model_registry.ModelVersionTag` instance to log.

        Returns:
            None
        """
        req_body = message_to_json(
            SetModelVersionTag(name=name, version=version, key=tag.key, value=tag.value)
        )
        self._call_endpoint(SetModelVersionTag, req_body)

    def delete_model_version_tag(self, name, version, key):
        """
        Delete a tag associated with the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key.

        Returns:
            None
        """
        req_body = message_to_json(DeleteModelVersionTag(name=name, version=version, key=key))
        self._call_endpoint(DeleteModelVersionTag, req_body)

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
        req_body = message_to_json(SetRegisteredModelAlias(name=name, alias=alias, version=version))
        self._call_endpoint(SetRegisteredModelAlias, req_body)

    def delete_registered_model_alias(self, name, alias):
        """
        Delete an alias associated with a registered model.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            None
        """
        req_body = message_to_json(DeleteRegisteredModelAlias(name=name, alias=alias))
        self._call_endpoint(DeleteRegisteredModelAlias, req_body)

    def get_model_version_by_alias(self, name, alias):
        """
        Get the model version instance by name and alias.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        req_body = message_to_json(GetModelVersionByAlias(name=name, alias=alias))
        response_proto = self._call_endpoint(GetModelVersionByAlias, req_body)
        return ModelVersion.from_proto(response_proto.model_version)
