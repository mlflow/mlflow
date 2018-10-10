from abc import abstractmethod, ABCMeta

from mlflow.entities import ViewType


class AbstractStore:
    """
    Abstract class for Backend Storage
    This class will define API interface for front ends to connect with various types of backends
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        Empty constructor for now. This is deliberately not marked as abstract, else every
        derived class would be forced to create one.
        """
        pass

    @abstractmethod
    def list_experiments(self, view_type=ViewType.ACTIVE_ONLY):
        """

        :param view_type: Qualify requested type of experiments.
        :return: a list of Experiment objects stored in store for requested view.
        """
        pass

    @abstractmethod
    def create_experiment(self, name, artifact_location):
        """
        Creates a new experiment.
        If an experiment with the given name already exists, throws exception.

        :param name: Desired name for an experiment
        :param artifact_location: Base location for artifacts in runs. May be None.
        :return: experiment_id (integer) for the newly created experiment if successful, else None
        """
        pass

    @abstractmethod
    def get_experiment(self, experiment_id):
        """
        Fetches the experiment by ID from the backend store.
        Throws an exception if experiment is not found or permanently deleted.

        :param experiment_id: Integer id for the experiment
        :return: A single Experiment object if it exists, otherwise raises an Exception.
        """
        pass

    @abstractmethod
    def delete_experiment(self, experiment_id):
        """
        Deletes the experiment from the backend store. Deleted experiments can be restored until
        permanently deleted.

        :param experiment_id: Integer id for the experiment
        """
        pass

    @abstractmethod
    def restore_experiment(self, experiment_id):
        """
        Restore deleted experiment unless it is permanently deleted.

        :param experiment_id: Integer id for the experiment
        """
        pass

    @abstractmethod
    def rename_experiment(self, experiment_id, new_name):
        """
        Update an experiment's name. The new name must be unique.

        :param experiment_id: Integer id for the experiment
        """
        pass

    @abstractmethod
    def get_run(self, run_uuid):
        """
        Fetches the run from backend store

        :param run_uuid: Unique identifier for the run
        :return: A single Run object if it exists, otherwise raises an Exception
        """
        pass

    def update_run_info(self, run_uuid, run_status, end_time):
        """
        Updates the metadata of the specified run.
        :return: RunInfo describing the updated run.
        """
        pass

    def create_run(self, experiment_id, user_id, run_name, source_type, source_name,
                   entry_point_name, start_time, source_version, tags, parent_run_id):
        """
        Creates a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        :param experiment_id: ID of the experiment for this run
        :param user_id: ID of the user launching this run
        :param source_type: Enum (integer) describing the source of the run
        :return: The created Run object
        """
        pass

    @abstractmethod
    def delete_run(self, run_id):
        """
        Deletes a run.
        :param run_id:
        """
        pass

    @abstractmethod
    def restore_run(self, run_id):
        """
        Restores a run.
        :param run_id:
        """
        pass

    def log_metric(self, run_uuid, metric):
        """
        Logs a metric for the specified run
        :param run_uuid: String id for the run
        :param metric: Metric instance to log
        """
        pass

    def log_param(self, run_uuid, param):
        """
        Logs a param for the specified run
        :param run_uuid: String id for the run
        :param param: Param instance to log
        """
        pass

    def set_tag(self, run_uuid, tag):
        """
        Sets a tag for the specified run
        :param run_uuid: String id for the run
        :param tag: RunTag instance to set
        """
        pass

    @abstractmethod
    def get_metric(self, run_uuid, metric_key):
        """
        Returns the last logged value for a given metric.

        :param run_uuid: Unique identifier for run
        :param metric_key: Metric name within the run

        :return: A single float value for the given metric if logged, else None
        """
        pass

    @abstractmethod
    def get_param(self, run_uuid, param_name):
        """
        Returns the value of the specified parameter.

        :param run_uuid: Unique identifier for run
        :param param_name: Parameter name within the run

        :return: Value of the given parameter if logged, else None
        """
        pass

    @abstractmethod
    def get_metric_history(self, run_uuid, metric_key):
        """
        Returns all logged value for a given metric.

        :param run_uuid: Unique identifier for run
        :param metric_key: Metric name within the run

        :return: A list of float values logged for the give metric if logged, else empty list
        """
        pass

    @abstractmethod
    def search_runs(self, experiment_ids, search_expressions, run_view_type):
        """
        Returns runs that match the given list of search expressions within the experiments.
        Given multiple search expressions, all these expressions are ANDed together for search.

        :param experiment_ids: List of experiment ids to scope the search
        :param search_expression: list of search expressions

        :return: A list of Run objects that satisfy the search expressions
        """
        pass

    @abstractmethod
    def list_run_infos(self, experiment_id, run_view_type):
        """
        Returns run information for runs which belong to the experiment_id

        :param experiment_id: The experiment id which to search.

        :return: A list of RunInfo objects that satisfy the search expressions
        """
        pass
