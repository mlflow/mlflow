from abc import ABCMeta, abstractmethod

from mlflow.utils.annotations import developer_stable


@developer_stable
class DefaultExperimentProvider:
    """
    Abstract base class for objects that provide the ID of an MLflow Experiment based on the
    current client context. For example, when the MLflow client is running in a Databricks Job,
    a provider is used to obtain the ID of the MLflow Experiment associated with the Job.

    Usually the experiment_id is set explicitly by the user, but if the experiment is not set,
    MLflow computes a default experiment id based on different contexts.
    When an experiment is created via the fluent ``mlflow.start_run`` method, MLflow iterates
    through the registered ``DefaultExperimentProvider``s until it finds one whose
    ``in_context()`` method returns ``True``; MLflow then calls the provider's
    ``get_experiment_id()`` method and uses the resulting experiment ID for Tracking operations.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def in_context(self):
        """Determine if the MLflow client is running in a context where this provider can
        identify an associated MLflow Experiment ID.

        Returns:
            True if the MLflow client is running in a context where the provider
            can identify an associated MLflow Experiment ID. False otherwise.

        """
        pass

    @abstractmethod
    def get_experiment_id(self):
        """Provide the MLflow Experiment ID for the current MLflow client context.

        Assumes that ``in_context()`` is ``True``.

        Returns:
            The ID of the MLflow Experiment associated with the current context.

        """
        pass
