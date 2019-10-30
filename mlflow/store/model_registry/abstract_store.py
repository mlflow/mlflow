from abc import abstractmethod, ABCMeta

from mlflow.entities.model_registry.model_version_stages import ALL_STAGES


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
    def update_registered_model(self, registered_model, new_name=None, description=None):
        """
        Updates metadata for RegisteredModel entity. Either ``new_name`` or ``description`` should
        be non-None. Backend raises exception if a registered model with given name does not exist.

        :param registered_model: :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        :param new_name: (Optional) New proposed name for the registered model.
        :param description: (Optional) New description.

        :return: A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        pass

    @abstractmethod
    def delete_registered_model(self, registered_model):
        """
        Delete registered model.
        Backend raises exception if a registered model with given name does not exist.

        :param registered_model: :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

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
    def get_registered_model_details(self, registered_model):
        """
        :param registered_model: :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        :return: A single :py:class:`mlflow.entities.model_registry.RegisteredModelDetailed` object.
        """
        pass

    @abstractmethod
    def get_latest_versions(self, registered_model, stages=None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        :param registered_model: :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        :param stages: List of desired stages. If input list is None, return latest versions for
                       for 'Staging' and 'Production' stages.

        :return: List of `:py:class:`mlflow.entities.model_registry.ModelVersionDetailed` objects.
        """
        pass

    # CRUD API for ModelVersion objects

    @abstractmethod
    def create_model_version(self, name, source, run_id):
        """
        Create a new model version from given source and run ID.

        :param name: Name ID for containing registered model.
        :param source: Source path where the MLflow model is stored.
        :param run_id: Run ID from MLflow tracking server that generated the model

        :return: A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
        created in the backend.
        """
        pass

    @abstractmethod
    def update_model_version(self, model_version, stage=None, description=None):
        """
        Update metadata associated with a model version in backend.

        :param model_version: :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        :param stage: New desired stage for this model version.
        :param description: New description.

        :return: None.
        """
        pass

    @abstractmethod
    def delete_model_version(self, model_version):
        """
        Delete model version in backend.

        :param model_version: :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        :return: None
        """
        pass

    @abstractmethod
    def get_model_version_details(self, model_version):
        """
        :param model_version: :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        :return: A single :py:class:`mlflow.entities.model_registry.ModelVersionDetailed` object.
        """
        pass

    @abstractmethod
    def get_model_version_download_uri(self, model_version):
        """
        Get the download location in Model Registry for this model version.

        :param model_version: :py:class:`mlflow.entities.model_registry.ModelVersion` object.

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

        :return: PagedList of :py:class:`mlflow.entities.model_registry.ModelVersionDetailed`
                 objects.
        """
        pass

    def get_model_version_stages(self, model_version):  # pylint: disable=unused-argument
        """
        Get all registry stages for the model

        :param model_version: :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        :return: A list of valid stages.
        """
        return ALL_STAGES
