from abc import ABCMeta, abstractmethod


class RunContextProvider(object):
    """
    Abstract base class for context provider objects specifying custom tags at run-creation time
    (e.g. tags specifying the git repo with which the run is associated).

    When a run is created via the fluent ``mlflow.start_run`` method, MLflow iterates through all
    registered RunContextProviders. For each context provider where ``in_context`` returns ``True``,
    MLflow calls the ``tags`` method on the context provider to compute context tags for the run.
    All context tags are then merged together and set on the newly-created run.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def in_context(self):
        """
        Determine if MLflow is running in this context.

        :return: bool indicating if in this context
        """
        raise NotImplementedError

    @abstractmethod
    def tags(self):
        """
        Generate context-specific tags.

        :return: dict of tags
        """
        raise NotImplementedError

    @abstractmethod
    def execute_start_run_actions(self, run):
        """
        Execute context-specific actions when a MLflow run is started

        :param run: An instance of :py:class:`mlflow.entities.Run` of the run started
        run that started
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def execute_end_run_actions(self, run, status):
        """
        Execute context-specific actions when a MLflow run is finished

        :param run: An instance of :py:class:`mlflow.entities.Run` of the run finished
        :param status: A string value of :py:class:`mlflow.entities.RunStatus`.
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def execute_create_experiment_actions(self, experiment_id):
        """
        Execute context-specific actions when a MLflow experiment is created

        :param experiment_id: Experiment ID of the created experiments.
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def execute_delete_experiment_actions(self, experiment_id):
        """
        Execute context-specific actions when a MLflow experiment is deleted

        :param experiment_id: Experiment ID of the deletd experiments.
        :return: None
        """
        raise NotImplementedError
