"""
Internal package providing a Python CRUD interface to MLflow models and versions.
This is a lower level API than the :py:mod:`mlflow.tracking.fluent` module, and is
exposed in the :py:mod:`mlflow.tracking` module.
"""

import logging

from mlflow.entities.model_registry import ModelVersionTag, RegisteredModelTag
from mlflow.exceptions import MlflowException
from mlflow.store.model_registry import (
    SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
    SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS, utils
from mlflow.utils.arguments_utils import _get_arg_names

_logger = logging.getLogger(__name__)


class ModelRegistryClient:
    """
    Client of an MLflow Model Registry Server that creates and manages registered
    models and model versions.
    """

    def __init__(self, registry_uri, tracking_uri):
        """
        Args:
            registry_uri: Address of local or remote model registry server.
            tracking_uri: Address of local or remote tracking server.
        """
        self.registry_uri = registry_uri
        self.tracking_uri = tracking_uri
        # NB: Fetch the tracking store (`self.store`) upon client initialization to ensure that
        # the tracking URI is valid and the store can be properly resolved. We define `store` as a
        # property method to ensure that the client is serializable, even if the store is not
        self.store

    @property
    def store(self):
        return utils._get_store(self.registry_uri, self.tracking_uri)

    # Registered Model Methods

    def create_registered_model(self, name, tags=None, description=None):
        """Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.model_registry.RegisteredModelTag` objects.
            description: Description of the model.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created by backend.

        """
        # TODO: Do we want to validate the name is legit here - non-empty without "/" and ":" ?
        #       Those are constraints applicable to any backend, given the model URI format.
        tags = tags if tags else {}
        tags = [RegisteredModelTag(key, str(value)) for key, value in tags.items()]
        return self.store.create_registered_model(name, tags, description)

    def update_registered_model(self, name, description):
        """Updates description for RegisteredModel entity.

        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Name of the registered model to update.
            description: New description.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
        return self.store.update_registered_model(name=name, description=description)

    def rename_registered_model(self, name, new_name):
        """Update registered model name.

        Args:
            name: Name of the registered model to update.
            new_name: New proposed name for the registered model.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
        if new_name.strip() == "":
            raise MlflowException("The name must not be an empty string.")
        return self.store.rename_registered_model(name=name, new_name=new_name)

    def delete_registered_model(self, name):
        """Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Name of the registered model to delete.
        """
        self.store.delete_registered_model(name)

    def search_registered_models(
        self,
        filter_string=None,
        max_results=SEARCH_REGISTERED_MODEL_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """Search for registered models in backend that satisfy the filter criteria.

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
        return self.store.search_registered_models(filter_string, max_results, order_by, page_token)

    def get_registered_model(self, name):
        """
        Args:
            name: Name of the registered model to get.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        return self.store.get_registered_model(name)

    def get_latest_versions(self, name, stages=None):
        """Latest version models for each requests stage. If no ``stages`` provided, returns the
        latest version for each stage.

        Args:
            name: Name of the registered model from which to get the latest versions.
            stages: List of desired stages. If input list is None, return latest versions for
                'Staging' and 'Production' stages.

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.

        """
        return self.store.get_latest_versions(name, stages)

    def set_registered_model_tag(self, name, key, value):
        """Set a tag for the registered model.

        Args:
            name: Registered model name.
            key: Tag key to log.
            value: Tag value log.

        Returns:
            None
        """
        self.store.set_registered_model_tag(name, RegisteredModelTag(key, str(value)))

    def delete_registered_model_tag(self, name, key):
        """Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
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
        local_model_path=None,
    ):
        """Create a new model version from given source.

        Args:
            name: Name of the containing registered model.
            source: URI indicating the location of the model artifacts.
            run_id: Run ID from MLflow tracking server that generated the model.
            tags: A dictionary of key-value pairs that are converted into
                :py:class:`mlflow.entities.model_registry.ModelVersionTag` objects.
            run_link: Link to the run from an MLflow tracking server that generated this model.
            description: Description of the version.
            await_creation_for: Number of seconds to wait for the model version to finish being
                created and is in ``READY`` status. By default, the function
                waits for five minutes. Specify 0 or None to skip waiting.
            local_model_path: Local path to the MLflow model, if it's already accessible on the
                local filesystem. Can be used by AbstractStores that upload model version files
                to the model registry to avoid a redundant download from the source location when
                logging and registering a model via a single
                mlflow.<flavor>.log_model(..., registered_model_name) call.

        Returns:
            Single :py:class:`mlflow.entities.model_registry.ModelVersion` object created by
            backend.

        """
        tags = tags if tags else {}
        tags = [ModelVersionTag(key, str(value)) for key, value in tags.items()]
        arg_names = _get_arg_names(self.store.create_model_version)
        if "local_model_path" in arg_names:
            mv = self.store.create_model_version(
                name,
                source,
                run_id,
                tags,
                run_link,
                description,
                local_model_path=local_model_path,
            )
        else:
            # Fall back to calling create_model_version without
            # local_model_path since old model registry store implementations may not
            # support the local_model_path argument.
            mv = self.store.create_model_version(name, source, run_id, tags, run_link, description)
        if await_creation_for and await_creation_for > 0:
            self.store._await_model_version_creation(mv, await_creation_for)
        return mv

    def copy_model_version(self, src_mv, dst_name):
        """Copy a model version from one registered model to another as a new model version.

        Args:
            src_mv: A :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
                the source model version.
            dst_name: The name of the registered model to copy the model version to. If a
                registered model with this name does not exist, it will be created.

        Returns:
            Single :py:class:`mlflow.entities.model_registry.ModelVersion` object representing
            the cloned model version.

        """
        return self.store.copy_model_version(src_mv=src_mv, dst_name=dst_name)

    def update_model_version(self, name, version, description):
        """Update metadata associated with a model version in backend.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.
            description: New description.
        """
        return self.store.update_model_version(name=name, version=version, description=description)

    def transition_model_version_stage(self, name, version, stage, archive_existing_versions=False):
        """Update model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automatically moved to the "archived" stage. Only
                valid when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will be
                raised.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

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
        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        return self.store.get_model_version(name, version)

    def delete_model_version(self, name, version):
        """Delete model version in backend.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.

        """
        self.store.delete_model_version(name, version)

    def get_model_version_download_uri(self, name, version):
        """Get the download location in Model Registry for this model version.

        Args:
            name: Name of the containing registered model.
            version: Version number of the model version.

        Returns:
            A single URI location that allows reads for downloading.

        """
        return self.store.get_model_version_download_uri(name, version)

    def search_model_versions(
        self,
        filter_string=None,
        max_results=SEARCH_MODEL_VERSION_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """Search for model versions in backend that satisfy the filter criteria.

        .. warning:

            The model version search results may not have aliases populated for performance reasons.

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
        return self.store.search_model_versions(filter_string, max_results, order_by, page_token)

    def get_model_version_stages(self, name, version):
        """
        Returns:
            A list of valid stages.
        """
        return self.store.get_model_version_stages(name, version)

    def set_model_version_tag(self, name, version, key, value):
        """Set a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key to log.
            value: Tag value to log.

        Returns:
            None
        """
        self.store.set_model_version_tag(name, version, ModelVersionTag(key, str(value)))

    def delete_model_version_tag(self, name, version, key):
        """Delete a tag associated with the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key.

        Returns:
            None
        """
        self.store.delete_model_version_tag(name, version, key)

    def set_registered_model_alias(self, name, alias, version):
        """Set a registered model alias pointing to a model version.

        Args:
            name: Registered model name.
            alias: Name of the alias.
            version: Registered model version number.

        Returns:
            None
        """
        self.store.set_registered_model_alias(name, alias, version)

    def delete_registered_model_alias(self, name, alias):
        """Delete an alias associated with a registered model.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            None
        """
        self.store.delete_registered_model_alias(name, alias)

    def get_model_version_by_alias(self, name, alias):
        """Get the model version instance by name and alias.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """
        return self.store.get_model_version_by_alias(name, alias)
