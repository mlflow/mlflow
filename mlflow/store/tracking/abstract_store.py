from abc import abstractmethod, ABCMeta

from mlflow.entities import ViewType
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT
from mlflow.utils import experimental


class AbstractStore:
    """
    Abstract class for Backend Storage.
    This class defines the API interface for front ends to connect with various types of backends.
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
        Create a new experiment.
        If an experiment with the given name already exists, throws exception.

        :param name: Desired name for an experiment
        :param artifact_location: Base location for artifacts in runs. May be None.

        :return: experiment_id (string) for the newly created experiment if successful, else None.
        """
        pass

    @abstractmethod
    def get_experiment(self, experiment_id):
        """
        Fetch the experiment by ID from the backend store.

        :param experiment_id: String id for the experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists,
            otherwise raises an exception.

        """
        pass

    def get_experiment_by_name(self, experiment_name):
        """
        Fetch the experiment by name from the backend store.
        This is a base implementation using ``list_experiments``, derived classes may have
        some specialized implementations.

        :param experiment_name: Name of experiment

        :return: A single :py:class:`mlflow.entities.Experiment` object if it exists.
        """
        for experiment in self.list_experiments(ViewType.ALL):
            if experiment.name == experiment_name:
                return experiment
        return None

    @abstractmethod
    def delete_experiment(self, experiment_id):
        """
        Delete the experiment from the backend store. Deleted experiments can be restored until
        permanently deleted.

        :param experiment_id: String id for the experiment
        """
        pass

    @abstractmethod
    def restore_experiment(self, experiment_id):
        """
        Restore deleted experiment unless it is permanently deleted.

        :param experiment_id: String id for the experiment
        """
        pass

    @abstractmethod
    def rename_experiment(self, experiment_id, new_name):
        """
        Update an experiment's name. The new name must be unique.

        :param experiment_id: String id for the experiment
        """
        pass

    @abstractmethod
    def get_run(self, run_id):
        """
        Fetch the run from backend store. The resulting :py:class:`Run <mlflow.entities.Run>`
        contains a collection of run metadata - :py:class:`RunInfo <mlflow.entities.RunInfo>`,
        as well as a collection of run parameters, tags, and metrics -
        :py:class`RunData <mlflow.entities.RunData>`. In the case where multiple metrics with the
        same key are logged for the run, the :py:class:`RunData <mlflow.entities.RunData>` contains
        the value at the latest timestamp for each metric. If there are multiple values with the
        latest timestamp for a given metric, the maximum of these values is returned.

        :param run_id: Unique identifier for the run.

        :return: A single :py:class:`mlflow.entities.Run` object, if the run exists. Otherwise,
                 raises an exception.
        """
        pass

    @abstractmethod
    def update_run_info(self, run_id, run_status, end_time):
        """
        Update the metadata of the specified run.

        :return: :py:class:`mlflow.entities.RunInfo` describing the updated run.
        """
        pass

    @abstractmethod
    def create_run(self, experiment_id, user_id, start_time, tags):
        """
        Create a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        :param experiment_id: String id of the experiment for this run
        :param user_id: ID of the user launching this run

        :return: The created Run object
        """
        pass

    @abstractmethod
    def delete_run(self, run_id):
        """
        Delete a run.

        :param run_id
        """
        pass

    @abstractmethod
    def restore_run(self, run_id):
        """
        Restore a run.

        :param run_id
        """
        pass

    def log_metric(self, run_id, metric):
        """
        Log a metric for the specified run

        :param run_id: String id for the run
        :param metric: :py:class:`mlflow.entities.Metric` instance to log
        """
        self.log_batch(run_id, metrics=[metric], params=[], tags=[])

    def log_param(self, run_id, param):
        """
        Log a param for the specified run

        :param run_id: String id for the run
        :param param: :py:class:`mlflow.entities.Param` instance to log
        """
        self.log_batch(run_id, metrics=[], params=[param], tags=[])

    def set_experiment_tag(self, experiment_id, tag):
        """
        Set a tag for the specified experiment

        :param experiment_id: String id for the experiment
        :param tag: :py:class:`mlflow.entities.ExperimentTag` instance to set
        """
        pass

    def set_tag(self, run_id, tag):
        """
        Set a tag for the specified run

        :param run_id: String id for the run
        :param tag: :py:class:`mlflow.entities.RunTag` instance to set
        """
        self.log_batch(run_id, metrics=[], params=[], tags=[tag])

    @abstractmethod
    def get_metric_history(self, run_id, metric_key):
        """
        Return a list of metric objects corresponding to all values logged for a given metric.

        :param run_id: Unique identifier for run
        :param metric_key: Metric name within the run

        :return: A list of :py:class:`mlflow.entities.Metric` entities if logged, else empty list
        """
        pass

    def search_runs(self, experiment_ids, filter_string, run_view_type,
                    max_results=SEARCH_MAX_RESULTS_DEFAULT, order_by=None, page_token=None):
        """
        Return runs that match the given list of search expressions within the experiments.

        :param page_token:
        :param page_token:
        :param experiment_ids: List of experiment ids to scope the search
        :param filter_string: A search filter string.
        :param run_view_type: ACTIVE_ONLY, DELETED_ONLY, or ALL runs
        :param max_results: Maximum number of runs desired.
        :param order_by: List of order_by clauses.
        :param page_token: Token specifying the next page of results. It should be obtained from
            a ``search_runs`` call.

        :return: A list of :py:class:`mlflow.entities.Run` objects that satisfy the search
            expressions. The pagination token for the next page can be obtained via the ``token``
            attribute of the object; however, some store implementations may not support pagination
            and thus the returned token would not be meaningful in such cases.
        """
        runs, token = self._search_runs(experiment_ids, filter_string, run_view_type, max_results,
                                        order_by, page_token)
        return PagedList(runs, token)

    @abstractmethod
    def _search_runs(self, experiment_ids, filter_string, run_view_type, max_results, order_by,
                     page_token):
        """
        Return runs that match the given list of search expressions within the experiments, as
        well as a pagination token (indicating where the next page should start). Subclasses of
        ``AbstractStore`` should implement this method to support pagination instead of
        ``search_runs``.

        See ``search_runs`` for parameter descriptions.

        :return: A tuple of ``runs`` and ``token`` where ``runs`` is a list of
            :py:class:`mlflow.entities.Run` objects that satisfy the search expressions,
            and ``token`` is the pagination token for the next page of results.
        """
        pass

    def list_run_infos(self, experiment_id, run_view_type):
        """
        Return run information for runs which belong to the experiment_id.

        :param experiment_id: The experiment id which to search

        :return: A list of :py:class:`mlflow.entities.RunInfo` objects that satisfy the
            search expressions
        """
        runs = self.search_runs([experiment_id], None, run_view_type)
        return [run.info for run in runs]

    @abstractmethod
    def log_batch(self, run_id, metrics, params, tags):
        """
        Log multiple metrics, params, and tags for the specified run

        :param run_id: String id for the run
        :param metrics: List of :py:class:`mlflow.entities.Metric` instances to log
        :param params: List of :py:class:`mlflow.entities.Param` instances to log
        :param tags: List of :py:class:`mlflow.entities.RunTag` instances to log

        :return: None.
        """
        pass

    @experimental
    @abstractmethod
    def record_logged_model(self, run_id, mlflow_model):
        """
        Record logged model information with tracking store. The list of logged model infos is
        maintained in a mlflow.models tag in JSON format.

        Note: The actual models are logged as artifacts via artifact repository.

        :param run_id: String id for the run
        :param mlflow_model: Model object to be recorded.

        NB: This API is experimental and may change in the future. The default implementation is a
        no-op.

        :return: None.
        """
        pass
