from abc import ABCMeta, abstractmethod


class DefaultExperimentProvider(object):
    """
    Abstract base class for context provider objects specifying experiment_id at run-creation time
    (e.g. experiment_id created for the job context for which the experiment is created).

    When an experiment is created via the fluent ``mlflow.start_run`` method,
    MLflow iterates through all registered DefaultExperimentProvider.
    For each context provider where ``in_context`` returns ``True``,
    MLflow calls the ``get_experiment_id`` method on the context provider
    to compute experiment_id for the experiment.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def in_context(self):
        """
        Determine if MLflow is running in this context.

        :return: bool indicating if in this context
        """
        pass

    @abstractmethod
    def get_experiment_id(self):
        """
        Generate context-specific experiment_id.

        :return: experiment_id
        """
        pass
