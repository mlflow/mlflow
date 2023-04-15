"""
Internal package providing a Python CRUD interface to MLflow models and versions.
This is a lower level API than the :py:mod:`mlflow.tracking.fluent` module, and is
exposed in the :py:mod:`mlflow.tracking` module.
"""
from datetime import timedelta, datetime
from time import sleep

import logging

from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
)
from mlflow.entities.model_registry import RegisteredModelTag, ModelVersionTag
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking._model_registry import utils, DEFAULT_AWAIT_MAX_SLEEP_SECONDS


_logger = logging.getLogger(__name__)

AWAIT_MODEL_VERSION_CREATE_SLEEP_DURATION_SECONDS = 3


class ModelRegistryClient:
    """
    Client of an MLflow Model Registry Server that creates and manages registered
    models and model versions.
    """

    def __init__(self, registry_uri, tracking_uri):
        """
        :param registry_uri: Address of local or remote model registry server.
        :param tracking_uri: Address of local or remote tracking server.
        """
        self.registry_uri = registry_uri
        self.tracking_uri = tracking_uri
        # NB: Fetch the tracking store (`self.store`) upon client initialization to ensure that
        # the tracking URI is valid and the store can be properly resolved. We define `store` as a
        # property method to ensure that the client is serializable, even if the store is not
        self.store  # pylint: disable=pointless-statement

    @property
    def store(self):
        return utils._get_store(self.registry_uri, self.tracking_uri)

    # Registered Model Methods

    def create_registered_model(self, name, tags=None, description=None):
        """
         Create a new registered model in backend store.

         :param name: Name of the new model. This is expected to be unique in the backend store.
         :param tags: A dictionary of key-value pairs that are converted into
                      :py:class:`mlflow.entities.model_registry.RegisteredModelTag` objects.
        :param description: Description of the model.
         :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
                  created by backend.
        """
        # TODO: Do we want to validate the name is legit here - non-empty without "/" and ":" ?
        #       Those are constraints applicable to any backend, given the model URI format.
        tags = tags if tags else {}
        tags = [RegisteredModelTag(key, str(value)) for key, value in tags.items()]
        return self.store.create_registered_model(name, tags, description)

    def update_registered_model(self, name, description):
        """
        Updates description for RegisteredModel entity.

        Backend raises exception if a registered model with given name does not exist.

        :param name: Name of the registered model to update.
        :param description: New description.
        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        return self.store.update_registered_model(name=name, description=description)

    def rename_registered_model(self, name, new_name):
        """
        Update registered model name.

        :param name: Name of the registered model to update.
        :param new_name: New proposed name for the registered model.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        if new_name.strip() == "":
            raise MlflowException("The name must not be an empty string.")
        return self.store.rename_registered_model(name=name, new_name=new_name)

    def delete_registered_model(self, name):
        """
        Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param name: Name of the registered model to delete.
        """
        self.store.delete_registered_model(name)

    def search_registered_models(
        self,
        filter_string=None,
        max_results=SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """
        Search for registered models in backend that satisfy the filter criteria.

        :param filter_string: Filter query string, defaults to searching all registered models.
        :param max_results: Maximum number of registered models desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_registered_models`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
                that satisfy the search expressions. The pagination token for the next page can be
                obtained via the ``token`` attribute of the object.
        """
        return self.store.search_registered_models(filter_string, max_results, order_by, page_token)

    def get_registered_model(self, name):
        """
        :param name: Name of the registered model to get.
        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        return self.store.get_registered_model(name)

    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requests stage. If no ``stages`` provided, returns the
        latest version for each stage.

        :param name: Name of the registered model from which to get the latest versions.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for 'Staging' and 'Production' stages.
        :return: List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        return self.store.get_latest_versions(name, stages)

    def set_registered_model_tag(self, name, key, value):
        """
        Set a tag for the registered model.

        :param name: Registered model name.
        :param key: Tag key to log.
        :param value: Tag value log.
        :return: None
        """
        self.store.set_registered_model_tag(name, RegisteredModelTag(key, str(value)))

    def delete_registered_model_tag(self, name, key):
        """
        Delete a tag associated with the registered model.

        :param name: Registered model name.
        :param key: Registered model tag key.
        :return: None
        """
        self.store.delete_registered_model_tag(name, key)

    # Model Version Methods

    def create_model_version(
        self,
        name,
        source,
        run_id=None,
        tags=None,
        run_link=None,
        description=None,
        await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    ):
        """
        Create a new model version from given source.

        :param name: Name of the containing registered model.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model.
        :param tags: A dictionary of key-value pairs that are converted into
                     :py:class:`mlflow.entities.model_registry.ModelVersionTag` objects.
        :param run_link: Link to the run from an MLflow tracking server that generated this model.
        :param description: Description of the version.
        :param await_creation_for: Number of seconds to wait for the model version to finish being
                                    created and is in ``READY`` status. By default, the function
                                    waits for five minutes. Specify 0 or None to skip waiting.
        Wait until the model version is finished being created and is in ``READY`` status.
        :return: Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
                 backend.
        """
        tags = tags if tags else {}
        tags = [ModelVersionTag(key, str(value)) for key, value in tags.items()]
        mv = self.store.create_model_version(name, source, run_id, tags, run_link, description)
        if await_creation_for and await_creation_for > 0:
            _logger.info(
                f"Waiting up to {await_creation_for} seconds for model version to finish creation. "
                f"Model name: {name}, version {mv.version}",
            )
            max_datetime = datetime.utcnow() + timedelta(seconds=await_creation_for)
            pending_status = ModelVersionStatus.to_string(ModelVersionStatus.PENDING_REGISTRATION)
            while mv.status == pending_status:
                if datetime.utcnow() > max_datetime:
                    raise MlflowException(
                        f"Exceeded max wait time for model name: {mv.name} version: {mv.version} "
                        f"to become READY. Status: {mv.status} Wait Time: {await_creation_for}"
                    )
                mv = self.get_model_version(mv.name, mv.version)
                sleep(AWAIT_MODEL_VERSION_CREATE_SLEEP_DURATION_SECONDS)
            if mv.status != ModelVersionStatus.to_string(ModelVersionStatus.READY):
                raise MlflowException(
                    f"Model version creation failed for model name: {mv.name} version: "
                    f"{mv.version} with status: {mv.status} and message: {mv.status_message}"
                )
        return mv

    def update_model_version(self, name, version, description):
        """
        Update metadata associated with a model version in backend.

        :param name: Name of the containing registered model.
        :param version: Version number of the model version.
        :param description: New description.
        """
        return self.store.update_model_version(name=name, version=version, description=description)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
        """
        Update model version stage.

        :param name: Registered model name.
        :param version: Registered model version.
        :param stage: New desired stage for this model version.
        :param archive_existing_versions: If this flag is set to ``True``, all existing model
            versions in the stage will be automatically moved to the "archived" stage. Only valid
            when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be raised.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        if stage.strip() == "":
            raise MlflowException("The stage must not be an empty string.")
        return self.store.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing_versions,
        )

    def get_model_version(self, name, version):
        """
        :param name: Name of the containing registered model.
        :param version: Version number of the model version.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        return self.store.get_model_version(name, version)

    def delete_model_version(self, name, version):
        """
        Delete model version in backend.

        :param name: Name of the containing registered model.
        :param version: Version number of the model version.
        """
        self.store.delete_model_version(name, version)

    def get_model_version_download_uri(self, name, version):
        """
        Get the download location in Model Registry for this model version.

        :param name: Name of the containing registered model.
        :param version: Version number of the model version.
        :return: A single URI location that allows reads for downloading.
        """
        return self.store.get_model_version_download_uri(name, version)

    def search_model_versions(
        self,
        filter_string=None,
        max_results=SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: A filter string expression. Currently supports a single filter
                              condition either name of model like ``name = 'model_name'`` or
                              ``run_id = '...'``.
        :param max_results: Maximum number of model versions desired.
        :param order_by: List of column names with ASC|DESC annotation, to be used for ordering
                         matching search results.
        :param page_token: Token specifying the next page of results. It should be obtained from
                            a ``search_model_versions`` call.
        :return: A PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 objects that satisfy the search expressions. The pagination token for the next
                 page can be obtained via the ``token`` attribute of the object.
        """
        return self.store.search_model_versions(filter_string, max_results, order_by, page_token)

    def get_model_version_stages(self, name, version):
        """
        :return: A list of valid stages.
        """
        return self.store.get_model_version_stages(name, version)

    def set_model_version_tag(self, name, version, key, value):
        """
        Set a tag for the model version.

        :param name: Registered model name.
        :param version: Registered model version.
        :param key: Tag key to log.
        :param value: Tag value to log.
        :return: None
        """
        self.store.set_model_version_tag(name, version, ModelVersionTag(key, str(value)))

    def delete_model_version_tag(self, name, version, key):
        """
        Delete a tag associated with the model version.

        :param name: Registered model name.
        :param version: Registered model version.
        :param key: Tag key.
        :return: None
        """
        self.store.delete_model_version_tag(name, version, key)

    def set_registered_model_alias(self, name, alias, version):
        """
        Set a registered model alias pointing to a model version.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :param version: Registered model version number.
        :return: None
        """
        self.store.set_registered_model_alias(name, alias, version)

    def delete_registered_model_alias(self, name, alias):
        """
        Delete an alias associated with a registered model.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :return: None
        """
        self.store.delete_registered_model_alias(name, alias)

    def get_model_version_by_alias(self, name, alias):
        """
        Get the model version instance by name and alias.

        :param name: Registered model name.
        :param alias: Name of the alias.
        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        return self.store.get_model_version_by_alias(name, alias)
