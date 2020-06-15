from abc import abstractmethod, ABCMeta


class AbstractStore:
    """
    Note:: Experimental: This entity may change or be removed in a future release without warning.
    Abstract class that defines API interfaces for storing Model Registry metadata.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Empty constructor. This is deliberately not marked as abstract, else every derived class
        would be forced to create one.
        """
        pass

    # CRUD API for RegisteredModel objects

    @abstractmethod
    def create_registered_model(self, name):
        """
        Create a new registered model in backend store.

        :param name: Name of the new model. This is expected to be unique in the backend store.

        :return: A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
        created in the backend.
        """
        pass

    @abstractmethod
    def update_registered_model(self, name, description):
        """
        Update description of the registered model.

        :param name: Registered model name.

        :param description: New description.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        pass

    @abstractmethod
    def rename_registered_model(self, name, new_name):
        """
        Rename the registered model.

        :param name: Registered model name.

        :param new_name: New proposed name.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        pass

    @abstractmethod
    def delete_registered_model(self, name):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param name: Registered model name.

        :return: None
        """
        pass

    @abstractmethod
    def list_registered_models(self):
        """
        List of all registered models.

        :return: List of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects.
        """
        pass

    @abstractmethod
    def get_registered_model(self, name):
        """
        :param name: Registered model name.

        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        pass

    @abstractmethod
    def get_latest_versions(self, name, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        :param name: Registered model name.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for 'Staging' and 'Production' stages.

        :return: List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        pass

    # CRUD API for ModelVersion objects

    @abstractmethod
    def create_model_version(self, name, source, run_id):
        """
        Create a new model version from given source and run ID.

        :param name: Registered model name.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model

        :return: A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
        created in the backend.
        """
        pass

    @abstractmethod
    def update_model_version(self, name, version, description):
        """
        Update metadata associated with a model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.
        :param description: New model description.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def delete_model_version(self, name, version):
        """
        Delete model version in backend.

        :param name: Registered model name.
        :param version: Registered model version.

        :return: None
        """
        pass

    @abstractmethod
    def get_model_version_download_uri(self, name, version):
        """
        Get the download location in Model Registry for this model version.

        :param name: Registered model name.
        :param version: Registered model version.

        :return: A single URI location that allows reads for downloading.
        """
        pass

    @abstractmethod
    def search_model_versions(self, filter_string):
        """
        Search for model versions in backend that satisfy the filter criteria.

        :param filter_string: A filter string expression. Currently supports a single filter
                              condition either name of model like ``name = 'model_name'`` or
                              ``run_id = '...'``.

        :return: PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
                 objects.
        """
        pass
