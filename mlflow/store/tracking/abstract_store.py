from abc import ABCMeta, abstractmethod
from typing import Optional

from mlflow.entities import (
    DatasetInput,
    TraceInfo,
    ViewType,
)
from mlflow.entities.metric import MetricWithRunId
from mlflow.entities.trace_status import TraceStatus
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.utils.annotations import developer_stable
from mlflow.utils.async_logging.async_logging_queue import AsyncLoggingQueue
from mlflow.utils.async_logging.run_operations import RunOperations


@developer_stable
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
        self._async_logging_queue = AsyncLoggingQueue(logging_func=self.log_batch)

    @abstractmethod
    def search_experiments(
        self,
        view_type=ViewType.ACTIVE_ONLY,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        filter_string=None,
        order_by=None,
        page_token=None,
    ):
        """
        Search for experiments that match the specified search query.

        Args:
            view_type: One of enum values ``ACTIVE_ONLY``, ``DELETED_ONLY``, or ``ALL``
                defined in :py:class:`mlflow.entities.ViewType`.
            max_results: Maximum number of experiments desired. Certain server backend may apply
                its own limit.
            filter_string: Filter query string (e.g., ``"name = 'my_experiment'"``), defaults to
                searching for all experiments. The following identifiers, comparators, and logical
                operators are supported.

                Identifiers
                  - ``name``: Experiment name
                  - ``creation_time``: Experiment creation time
                  - ``last_update_time``: Experiment last update time
                  - ``tags.<tag_key>``: Experiment tag. If ``tag_key`` contains
                    spaces, it must be wrapped with backticks (e.g., ``"tags.`extra key`"``).

                Comparators for string attributes and tags
                  - ``=``: Equal to
                  - ``!=``: Not equal to
                  - ``LIKE``: Case-sensitive pattern match
                  - ``ILIKE``: Case-insensitive pattern match

                Comparators for numeric attributes
                  - ``=``: Equal to
                  - ``!=``: Not equal to
                  - ``<``: Less than
                  - ``<=``: Less than or equal to
                  - ``>``: Greater than
                  - ``>=``: Greater than or equal to

                Logical operators
                  - ``AND``: Combines two sub-queries and returns True if both of them are True.

            order_by: List of columns to order by. The ``order_by`` column can contain an optional
                ``DESC`` or ``ASC`` value (e.g., ``"name DESC"``). The default ordering is ``ASC``,
                so ``"name"`` is equivalent to ``"name ASC"``. If unspecified, defaults to
                ``["last_update_time DESC"]``, which lists experiments updated most recently first.
                The following fields are supported:

                - ``experiment_id``: Experiment ID
                - ``name``: Experiment name
                - ``creation_time``: Experiment creation time
                - ``last_update_time``: Experiment last update time

            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_experiments`` call.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Experiment <mlflow.entities.Experiment>` objects. The pagination token
            for the next page can be obtained via the ``token`` attribute of the object.

        """

    @abstractmethod
    def create_experiment(self, name, artifact_location, tags):
        """
        Create a new experiment.
        If an experiment with the given name already exists, throws exception.

        Args:
            name: Desired name for an experiment.
            artifact_location: Base location for artifacts in runs. May be None.
            tags: Experiment tags to set upon experiment creation

        Returns:
            experiment_id (string) for the newly created experiment if successful, else None.

        """

    @abstractmethod
    def get_experiment(self, experiment_id):
        """
        Fetch the experiment by ID from the backend store.

        Args:
            experiment_id: String id for the experiment

        Returns:
            A single :py:class:`mlflow.entities.Experiment` object if it exists,
            otherwise raises an exception.
        """

    def get_experiment_by_name(self, experiment_name):
        """
        Fetch the experiment by name from the backend store.

        Args:
            experiment_name: Name of experiment

        Returns:
            A single :py:class:`mlflow.entities.Experiment` object if it exists.
        """

    @abstractmethod
    def delete_experiment(self, experiment_id):
        """
        Delete the experiment from the backend store. Deleted experiments can be restored until
        permanently deleted.

        Args:
            experiment_id: String id for the experiment.
        """

    @abstractmethod
    def restore_experiment(self, experiment_id):
        """
        Restore deleted experiment unless it is permanently deleted.

        Args:
            experiment_id: String id for the experiment.
        """

    @abstractmethod
    def rename_experiment(self, experiment_id, new_name):
        """
        Update an experiment's name. The new name must be unique.

        Args:
            experiment_id: String id for the experiment.
            new_name: New name for the experiment.
        """

    @abstractmethod
    def get_run(self, run_id):
        """
        Fetch the run from backend store. The resulting :py:class:`Run <mlflow.entities.Run>`
        contains a collection of run metadata - :py:class:`RunInfo <mlflow.entities.RunInfo>`,
        as well as a collection of run parameters, tags, and metrics -
        :py:class:`RunData <mlflow.entities.RunData>`. In the case where multiple metrics with the
        same key are logged for the run, the :py:class:`RunData <mlflow.entities.RunData>` contains
        the value at the latest timestamp for each metric. If there are multiple values with the
        latest timestamp for a given metric, the maximum of these values is returned.

        Args:
            run_id: Unique identifier for the run.

        Returns:
            A single :py:class:`mlflow.entities.Run` object, if the run exists. Otherwise,
            raises an exception.
        """

    @abstractmethod
    def update_run_info(self, run_id, run_status, end_time, run_name):
        """
        Update the metadata of the specified run.

        Returns:
            mlflow.entities.RunInfo: Describing the updated run.
        """

    @abstractmethod
    def create_run(self, experiment_id, user_id, start_time, tags, run_name):
        """
        Create a run under the specified experiment ID, setting the run's status to "RUNNING"
        and the start time to the current time.

        Args:
            experiment_id: String id of the experiment for this run.
            user_id: ID of the user launching this run.
            start_time: Start time of the run.
            tags: A dictionary of string keys and string values.
            run_name: Name of the run.

        Returns:
            The created Run object
        """

    @abstractmethod
    def delete_run(self, run_id):
        """
        Delete a run.

        Args:
            run_id: The ID of the run to delete.

        """

    @abstractmethod
    def restore_run(self, run_id):
        """
        Restore a run.

        Args:
            run_id: The ID of the run to restore.

        """

    # TODO: rename this to create_trace_info
    def start_trace(
        self,
        experiment_id: str,
        timestamp_ms: int,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ) -> TraceInfo:
        """
        Start an initial TraceInfo object in the backend store.

        Args:
            experiment_id: String id of the experiment for this run.
            timestamp_ms: Start time of the trace, in milliseconds since the UNIX epoch.
            request_metadata: Metadata of the trace.
            tags: Tags of the trace.

        Returns:
            The created TraceInfo object.
        """
        raise NotImplementedError

    # TODO: rename this to update_trace_info
    # can we pass in execution_time_ms instead of timestamp_ms directly?
    def end_trace(
        self,
        request_id: str,
        timestamp_ms: int,
        status: TraceStatus,
        request_metadata: dict[str, str],
        tags: dict[str, str],
    ) -> TraceInfo:
        """
        Update the TraceInfo object in the backend store with the completed trace info.

        Args:
            request_id : Unique string identifier of the trace.
            timestamp_ms: End time of the trace, in milliseconds. The execution time field
                in the TraceInfo will be calculated by subtracting the start time from this.
            status: Status of the trace.
            request_metadata: Metadata of the trace. This will be merged with the existing
                metadata logged during the start_trace call.
            tags: Tags of the trace. This will be merged with the existing tags logged
                during the start_trace or set_trace_tag calls.

        Returns:
            The updated TraceInfo object.
        """
        raise NotImplementedError

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: Optional[int] = None,
        max_traces: Optional[int] = None,
        request_ids: Optional[list[str]] = None,
    ) -> int:
        """
        Delete traces based on the specified criteria.

        - Either `max_timestamp_millis` or `request_ids` must be specified, but not both.
        - `max_traces` can't be specified if `request_ids` is specified.

        Args:
            experiment_id: ID of the associated experiment.
            max_timestamp_millis: The maximum timestamp in milliseconds since the UNIX epoch for
                deleting traces. Traces older than or equal to this timestamp will be deleted.
            max_traces: The maximum number of traces to delete. If max_traces is specified, and
                it is less than the number of traces that would be deleted based on the
                max_timestamp_millis, the oldest traces will be deleted first.
            request_ids: A set of request IDs to delete.

        Returns:
            The number of traces deleted.
        """
        # request_ids can't be an empty list of string
        if max_timestamp_millis is None and not request_ids:
            raise MlflowException.invalid_parameter_value(
                "Either `max_timestamp_millis` or `request_ids` must be specified.",
            )
        if max_timestamp_millis and request_ids:
            raise MlflowException.invalid_parameter_value(
                "Only one of `max_timestamp_millis` and `request_ids` can be specified.",
            )
        if request_ids and max_traces is not None:
            raise MlflowException.invalid_parameter_value(
                "`max_traces` can't be specified if `request_ids` is specified.",
            )
        if max_traces is not None and max_traces <= 0:
            raise MlflowException.invalid_parameter_value(
                f"`max_traces` must be a positive integer, received {max_traces}.",
            )
        return self._delete_traces(experiment_id, max_timestamp_millis, max_traces, request_ids)

    def _delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: Optional[int] = None,
        max_traces: Optional[int] = None,
        request_ids: Optional[list[str]] = None,
    ) -> int:
        raise NotImplementedError

    def get_trace_info(self, request_id: str) -> TraceInfo:
        """
        Get the trace matching the `request_id`.

        Args:
            request_id: String id of the trace to fetch.

        Returns:
            The fetched Trace object, of type ``mlflow.entities.TraceInfo``.
        """
        raise NotImplementedError

    def search_traces(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str] = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: Optional[list[str]] = None,
        page_token: Optional[str] = None,
    ) -> tuple[list[TraceInfo], Optional[str]]:
        """
        Return traces that match the given list of search expressions within the experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search.
            filter_string: A search filter string.
            max_results: Maximum number of traces desired.
            order_by: List of order_by clauses.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_traces`` call.

        Returns:
            A tuple of a list of :py:class:`TraceInfo <mlflow.entities.TraceInfo>` objects that
            satisfy the search expressions and a pagination token for the next page of results.
            If the underlying tracking store supports pagination, the token for the
            next page may be obtained via the ``token`` attribute of the returned object; however,
            some store implementations may not support pagination and thus the returned token would
            not be meaningful in such cases.
        """
        raise NotImplementedError

    def set_trace_tag(self, request_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given request_id.

        Args:
            request_id: The ID of the trace.
            key: The string key of the tag.
            value: The string value of the tag.
        """
        raise NotImplementedError

    def delete_trace_tag(self, request_id: str, key: str):
        """
        Delete a tag on the trace with the given request_id.

        Args:
            request_id: The ID of the trace.
            key: The string key of the tag.
        """
        raise NotImplementedError

    def log_metric(self, run_id, metric):
        """
        Log a metric for the specified run

        Args:
            run_id: String id for the run
            metric: `mlflow.entities.Metric` instance to log
        """
        self.log_batch(run_id, metrics=[metric], params=[], tags=[])

    def log_metric_async(self, run_id, metric) -> RunOperations:
        """
        Log a metric for the specified run in async fashion.

        Args:
            run_id: String id for the run
            metric: `mlflow.entities.Metric` instance to log
        """
        return self.log_batch_async(run_id, metrics=[metric], params=[], tags=[])

    def log_param(self, run_id, param):
        """
        Log a param for the specified run

        Args:
            run_id: String id for the run
            param: :py:class:`mlflow.entities.Param` instance to log
        """
        self.log_batch(run_id, metrics=[], params=[param], tags=[])

    def log_param_async(self, run_id, param) -> RunOperations:
        """
        Log a param for the specified run in async fashion.

        Args:
            run_id: String id for the run.
            param: :py:class:`mlflow.entities.Param` instance to log.
        """
        return self.log_batch_async(run_id, metrics=[], params=[param], tags=[])

    def set_experiment_tag(self, experiment_id, tag):
        """
        Set a tag for the specified experiment

        Args:
            experiment_id: String id for the experiment.
            tag: :py:class:`mlflow.entities.ExperimentTag` instance to set.
        """

    def set_tag(self, run_id, tag):
        """
        Set a tag for the specified run

        Args:
            run_id: String id for the run.
            tag: :py:class:`mlflow.entities.RunTag` instance to set.
        """
        self.log_batch(run_id, metrics=[], params=[], tags=[tag])

    def set_tag_async(self, run_id, tag) -> RunOperations:
        """
        Set a tag for the specified run in async fashion.

        Args:
            run_id: String id for the run.
            tag: :py:class:`mlflow.entities.RunTag` instance to set.
        """
        return self.log_batch_async(run_id, metrics=[], params=[], tags=[tag])

    @abstractmethod
    def get_metric_history(self, run_id, metric_key, max_results=None, page_token=None):
        """
        Return a list of metric objects corresponding to all values logged for a given metric
        within a run.

        Args:
            run_id: Unique identifier for run.
            metric_key: Metric name within the run.
            max_results: Maximum number of metric history events (steps) to return per paged
                query.
            page_token: A Token specifying the next paginated set of results of metric history.
                This value is obtained as a return value from a paginated call to GetMetricHistory.

        Returns:
            A list of :py:class:`mlflow.entities.Metric` entities if logged, else empty list.
        """

        # NB: Pagination for this API is not supported in FileStore or SQLAlchemyStore. The
        # argument `max_results` is used as a pagination activation flag. If the `max_results`
        # argument is not provided, this API will return a full metric history event collection
        # without the paged queries to the backend store.

    def get_metric_history_bulk_interval_from_steps(self, run_id, metric_key, steps, max_results):
        """
        Return a list of metric objects corresponding to all values logged
        for a given metric within a run for the specified steps.

        Args:
            run_id: Unique identifier for run.
            metric_key: Metric name within the run.
            steps: List of steps for which to return metrics.
            max_results: Maximum number of metric history events (steps) to return.

        Returns:
            A list of MetricWithRunId objects:
                - key: Metric name within the run.
                - value: Metric value.
                - timestamp: Metric timestamp.
                - step: Metric step.
                - run_id: Unique identifier for run.
        """
        metrics_for_run = sorted(
            [m for m in self.get_metric_history(run_id, metric_key) if m.step in steps],
            key=lambda metric: (metric.step, metric.timestamp),
        )[:max_results]
        return [
            MetricWithRunId(
                run_id=run_id,
                metric=metric,
            )
            for metric in metrics_for_run
        ]

    def search_runs(
        self,
        experiment_ids,
        filter_string,
        run_view_type,
        max_results=SEARCH_MAX_RESULTS_DEFAULT,
        order_by=None,
        page_token=None,
    ):
        """
        Return runs that match the given list of search expressions within the experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search.
            filter_string: A search filter string.
            run_view_type: ACTIVE_ONLY, DELETED_ONLY, or ALL runs.
            max_results: Maximum number of runs desired.
            order_by: List of order_by clauses.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_runs`` call.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`Run <mlflow.entities.Run>` objects that satisfy the search expressions.
            If the underlying tracking store supports pagination, the token for the next page may
            be obtained via the ``token`` attribute of the returned object; however, some store
            implementations may not support pagination and thus the returned token would not be
            meaningful in such cases.
        """
        runs, token = self._search_runs(
            experiment_ids,
            filter_string,
            run_view_type,
            max_results,
            order_by,
            page_token,
        )
        return PagedList(runs, token)

    @abstractmethod
    def _search_runs(
        self,
        experiment_ids,
        filter_string,
        run_view_type,
        max_results,
        order_by,
        page_token,
    ):
        """
        Return runs that match the given list of search expressions within the experiments, as
        well as a pagination token (indicating where the next page should start). Subclasses of
        ``AbstractStore`` should implement this method to support pagination instead of
        ``search_runs``.

        See ``search_runs`` for parameter descriptions.

        Returns:
            A tuple of ``runs`` and ``token`` where ``runs`` is a list of
            :py:class:`mlflow.entities.Run` objects that satisfy the search expressions,
            and ``token`` is the pagination token for the next page of results.
        """

    @abstractmethod
    def log_batch(self, run_id, metrics, params, tags):
        """
        Log multiple metrics, params, and tags for the specified run

        Args:
            run_id: String id for the run
            metrics: List of :py:class:`mlflow.entities.Metric` instances to log
            params: List of :py:class:`mlflow.entities.Param` instances to log
            tags: List of :py:class:`mlflow.entities.RunTag` instances to log

        Returns:
            None.
        """

    def log_batch_async(self, run_id, metrics, params, tags) -> RunOperations:
        """
        Log multiple metrics, params, and tags for the specified run in async fashion.
        This API does not offer immediate consistency of the data. When API returns,
        data is accepted but not persisted/processed by back end. Data would be processed
        in near real time fashion.

        Args:
            run_id: String id for the run.
            metrics: List of :py:class:`mlflow.entities.Metric` instances to log.
            params: List of :py:class:`mlflow.entities.Param` instances to log.
            tags: List of :py:class:`mlflow.entities.RunTag` instances to log.

        Returns:
            An :py:class:`mlflow.utils.async_logging.run_operations.RunOperations` instance
            that represents future for logging operation.
        """
        if not self._async_logging_queue.is_active():
            self._async_logging_queue.activate()

        return self._async_logging_queue.log_batch_async(
            run_id=run_id, metrics=metrics, params=params, tags=tags
        )

    def end_async_logging(self):
        """
        Ends the async logging queue. This method is a no-op if the queue is not active. This is
        different from flush as it just stops the async logging queue from accepting
        new data (moving the queue state TEAR_DOWN state), but flush will ensure all data
        is processed before returning (moving the queue to IDLE state).
        """
        if self._async_logging_queue.is_active():
            self._async_logging_queue.end_async_logging()

    def flush_async_logging(self):
        """
        Flushes the async logging queue. This method is a no-op if the queue is already
        at IDLE state. This methods also shutdown the logging worker threads.
        After flushing, logging thread is setup again.
        """
        if not self._async_logging_queue.is_idle():
            self._async_logging_queue.flush()

    def shut_down_async_logging(self):
        """
        Shuts down the async logging queue. This method is a no-op if the queue is already
        at IDLE state. This methods also shutdown the logging worker threads.
        """
        if not self._async_logging_queue.is_idle():
            self._async_logging_queue.shut_down_async_logging()

    @abstractmethod
    def record_logged_model(self, run_id, mlflow_model):
        """
        Record logged model information with tracking store. The list of logged model infos is
        maintained in a mlflow.models tag in JSON format.

        Note: The actual models are logged as artifacts via artifact repository.

        Args:
            run_id: String id for the run.
            mlflow_model: Model object to be recorded.

        The default implementation is a no-op.

        Returns:
            None.
        """

    @abstractmethod
    def log_inputs(self, run_id: str, datasets: Optional[list[DatasetInput]] = None):
        """
        Log inputs, such as datasets, to the specified run.

        Args:
            run_id: String id for the run
            datasets: List of :py:class:`mlflow.entities.DatasetInput` instances to log
                as inputs to the run.

        Returns:
            None.
        """
