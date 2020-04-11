"""
Internal package providing a Python CRUD interface to MLflow models and versions.
This is a lower level API than the :py:mod:`mlflow.tracking.fluent` module, and is
exposed in the :py:mod:`mlflow.tracking` module.
"""

import logging

from mlflow.exceptions import MlflowException
from mlflow.tracking._model_registry import utils

_logger = logging.getLogger(__name__)


class ModelRegistryClient(object):
    """
    Client of an MLflow Model Registry Server that creates and manages registered
    models and model versions.
    """

    def __init__(self, registry_uri):
        """
        :param registry_uri: Address of local or remote model registry server.
        """
        self.registry_uri = registry_uri
        self.store = utils._get_store(self.registry_uri)

    # Registered Model Methods

    def create_registered_model(self, name):
        """
        Create a new registered model in backend store.

        :param name: Name of the new model. This is expected to be unique in the backend store.
        :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
                 created by backend.
        """
        # TODO: Do we want to validate the name is legit here - non-empty without "/" and ":" ?
        #       Those are constraints applicable to any backend, given the model URI format.
        return self.store.create_registered_model(name)

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

        :param name: Name of the registered model to update.
        """
        self.store.delete_registered_model(name)

    def list_registered_models(self):
        """
        List of all registered models.

        :return: List of :py:class:`mlflow.entities.registry.RegisteredModel` objects.
        """
        return self.store.list_registered_models()

    def get_registered_model(self, name):
        """
        :param name: Name of the registered model to update.
        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        return self.store.get_registered_model(name)

    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requests stage. If no ``stages`` provided, returns the
        latest version for each stage.

        :param name: Name of the registered model to update.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for 'Staging' and 'Production' stages.
        :return: List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        return self.store.get_latest_versions(name, stages)

    # Model Version Methods

    def create_model_version(self, name, source, run_id):
        """
        Create a new model version from given source or run ID.

        :param name: Name ID for containing registered model.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model
        :return: Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
                 backend.
        """
        return self.store.create_model_version(name, source, run_id)

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
        :param archive_existing_versions: If this flag is set, all existing model
               versions in the stage will be atomically moved to the "archived" stage.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        if stage.strip() == "":
            raise MlflowException("The stage must not be an empty string.")
        return self.store.transition_model_version_stage(
            name=name, version=version, stage=stage,
            archive_existing_versions=archive_existing_versions)

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

    def search_model_versions(self, filter_string):
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: A filter string expression. Currently supports a single filter
                              condition either name of model like ``name = 'model_name'`` or
                              ``run_id = '...'``.
        :return: PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        return self.store.search_model_versions(filter_string)

    def get_model_version_stages(self, name, version):
        """
        :return: A list of valid stages.
        """
        return self.store.get_model_version_stages(name, version)
