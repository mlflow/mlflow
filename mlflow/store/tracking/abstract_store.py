import json
from abc import ABCMeta, abstractmethod
from typing import Any

from mlflow.entities import (
    Assessment,
    DatasetInput,
    LoggedModel,
    LoggedModelInput,
    LoggedModelOutput,
    LoggedModelParameter,
    LoggedModelStatus,
    LoggedModelTag,
    ViewType,
)
from mlflow.entities.metric import MetricWithRunId
from mlflow.entities.trace import Span
from mlflow.entities.trace_info import TraceInfo
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import SEARCH_MAX_RESULTS_DEFAULT, SEARCH_TRACES_DEFAULT_MAX_RESULTS
from mlflow.utils import mlflow_tags
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

    def start_trace(self, trace_info: TraceInfo) -> TraceInfo:
        """
        Create a trace using the V3 API format with a complete Trace object.

        Args:
            trace_info: The TraceInfo object to create in the backend.

        Returns:
            The returned TraceInfo object from the backend.
        """
        raise NotImplementedError

    def delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        """
        Delete traces based on the specified criteria.

        - Either `max_timestamp_millis` or `trace_ids` must be specified, but not both.
        - `max_traces` can't be specified if `trace_ids` is specified.

        Args:
            experiment_id: ID of the associated experiment.
            max_timestamp_millis: The maximum timestamp in milliseconds since the UNIX epoch for
                deleting traces. Traces older than or equal to this timestamp will be deleted.
            max_traces: The maximum number of traces to delete. If max_traces is specified, and
                it is less than the number of traces that would be deleted based on the
                max_timestamp_millis, the oldest traces will be deleted first.
            trace_ids: A set of trace IDs to delete.

        Returns:
            The number of traces deleted.
        """
        # trace_ids can't be an empty list of string
        if max_timestamp_millis is None and not trace_ids:
            raise MlflowException.invalid_parameter_value(
                "Either `max_timestamp_millis` or `trace_ids` must be specified.",
            )
        if max_timestamp_millis and trace_ids:
            raise MlflowException.invalid_parameter_value(
                "Only one of `max_timestamp_millis` and `trace_ids` can be specified.",
            )
        if trace_ids and max_traces is not None:
            raise MlflowException.invalid_parameter_value(
                "`max_traces` can't be specified if `trace_ids` is specified.",
            )
        if max_traces is not None and max_traces <= 0:
            raise MlflowException.invalid_parameter_value(
                f"`max_traces` must be a positive integer, received {max_traces}.",
            )
        return self._delete_traces(experiment_id, max_timestamp_millis, max_traces, trace_ids)

    def _delete_traces(
        self,
        experiment_id: str,
        max_timestamp_millis: int | None = None,
        max_traces: int | None = None,
        trace_ids: list[str] | None = None,
    ) -> int:
        raise NotImplementedError

    def get_trace_info(self, trace_id: str) -> TraceInfo:
        """
        Get the trace matching the `trace_id`.

        Args:
            trace_id: String id of the trace to fetch.

        Returns:
            The fetched Trace object, of type ``mlflow.entities.TraceInfo``.
        """
        raise NotImplementedError

    def get_online_trace_details(
        self,
        trace_id: str,
        sql_warehouse_id: str,
        source_inference_table: str,
        source_databricks_request_id: str,
    ) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support `get_online_trace_details`."
        )

    def search_traces(
        self,
        experiment_ids: list[str],
        filter_string: str | None = None,
        max_results: int = SEARCH_TRACES_DEFAULT_MAX_RESULTS,
        order_by: list[str] | None = None,
        page_token: str | None = None,
        model_id: str | None = None,
        sql_warehouse_id: str | None = None,
    ) -> tuple[list[TraceInfo], str | None]:
        """
        Return traces that match the given list of search expressions within the experiments.

        Args:
            experiment_ids: List of experiment ids to scope the search.
            filter_string: A search filter string.
            max_results: Maximum number of traces desired.
            order_by: List of order_by clauses.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_traces`` call.
            model_id: If specified, return traces associated with the model ID.
            sql_warehouse_id: Only used in Databricks. The ID of the SQL warehouse to use for
                searching traces in inference tables.

        Returns:
            A tuple of a list of :py:class:`TraceInfo <mlflow.entities.TraceInfo>` objects that
            satisfy the search expressions and a pagination token for the next page of results.
            If the underlying tracking store supports pagination, the token for the
            next page may be obtained via the ``token`` attribute of the returned object; however,
            some store implementations may not support pagination and thus the returned token would
            not be meaningful in such cases.
        """
        raise NotImplementedError

    def set_trace_tag(self, trace_id: str, key: str, value: str):
        """
        Set a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
            value: The string value of the tag.
        """
        raise NotImplementedError

    def delete_trace_tag(self, trace_id: str, key: str):
        """
        Delete a tag on the trace with the given trace_id.

        Args:
            trace_id: The ID of the trace.
            key: The string key of the tag.
        """
        raise NotImplementedError

    def get_assessment(self, trace_id: str, assessment_id: str) -> Assessment:
        """
        Retrieve an assessment from a given trace.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The assessment identifier that denotes a unique assessment entry
                for a given trace.

        Returns:
            The Assessment object for the given trace and assessment ids.
        """
        raise NotImplementedError

    def create_assessment(self, assessment: Assessment) -> Assessment:
        """
        Logs an Assessment for a given trace or a span within a trace.

        Args:
            assessment: An :py:class:`Assessment <mlflow.entities.Assessment>` object that
                contains the key value mappings of assessment criteria comprised of either
                expectations or user/system/scorer-provided feedback (label data) on the quality
                of the trace response or for a span within a trace.

        Returns:
            The Assessment object for the logging operation.
        """
        raise NotImplementedError

    def update_assessment(
        self,
        trace_id: str,
        assessment_id: str,
        name: str | None = None,
        expectation: str | None = None,
        feedback: str | None = None,
        rationale: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Assessment:
        """
        Updates the given Assessment's mutable values to overwrite updated values
        for the given trace and Assessment data.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The ID of the assessment upon which overrides will be applied to
                mutable attributes.
            name: An Optional override to the name of the assessment.
            expectation: An Optional override of the expectation for the assessment.
            feedback: An Optional override to the feedback for a given assessment.
            rationale: An Optional string defining the reasoning behind the override of
                the assessment.
            metadata: An Optional mapping of additional customizable metadata for the assessment.

        Returns:
            The Assessment object representing the updated state of an assessment for a given trace.
        """
        raise NotImplementedError

    def delete_assessment(self, trace_id: str, assessment_id):
        """
        Delete an assessment for a given trace.

        Args:
            trace_id: The ID of the trace.
            assessment_id: The ID of the assessment to be deleted.
        """
        raise NotImplementedError

    def log_spans(self, experiment_id: str, spans: list[Span]) -> list[Span]:
        """
        Log multiple span entities to the tracking store.

        Args:
            experiment_id: The experiment ID to log spans to.
            spans: List of Span entities to log. All spans must belong to the same trace.

        Returns:
            List of logged Span entities.

        Raises:
            MlflowException: If spans belong to different traces.
        """
        raise NotImplementedError

    async def log_spans_async(self, experiment_id: str, spans: list[Span]) -> list[Span]:
        """
        Asynchronously log multiple span entities to the tracking store.

        Args:
            experiment_id: The experiment ID to log spans to.
            spans: List of Span entities to log. All spans must belong to the same trace.

        Returns:
            List of logged Span entities.

        Raises:
            MlflowException: If spans belong to different traces.
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

    def delete_experiment_tag(self, experiment_id, key):
        """
        Delete a tag from the specified experiment

        Args:
            experiment_id: String id for the experiment.
            key: String name of the tag to be deleted.
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
    def log_inputs(
        self,
        run_id: str,
        datasets: list[DatasetInput] | None = None,
        models: list[LoggedModelInput] | None = None,
    ):
        """
        Log inputs, such as datasets, to the specified run.

        Args:
            run_id: String id for the run
            datasets: List of :py:class:`mlflow.entities.DatasetInput` instances to log
                as inputs to the run.
            models: List of :py:class:`mlflow.entities.LoggedModelInput` instances to log
                as inputs to the run.

        Returns:
            None.
        """

    def log_outputs(self, run_id: str, models: list[LoggedModelOutput]):
        """
        Log outputs, such as models, to the specified run.

        Args:
            run_id: String id for the run
            models: List of :py:class:`mlflow.entities.LoggedModelOutput` instances to log
                as outputs of the run.

        Returns:
            None.
        """
        raise NotImplementedError(self.__class__.__name__)

    def record_logged_model(self, run_id, mlflow_model):
        raise NotImplementedError(self.__class__.__name__)

    def create_logged_model(
        self,
        experiment_id: str,
        name: str | None = None,
        source_run_id: str | None = None,
        tags: list[LoggedModelTag] | None = None,
        params: list[LoggedModelParameter] | None = None,
        model_type: str | None = None,
    ) -> LoggedModel:
        """
        Create a new logged model.

        Args:
            experiment_id: ID of the experiment to which the model belongs.
            name: Name of the model. If not specified, a random name will be generated.
            source_run_id: ID of the run that produced the model.
            tags: Tags to set on the model.
            params: Parameters to set on the model.
            model_type: Type of the model.

        Returns:
            The created model.
        """
        raise NotImplementedError(self.__class__.__name__)

    def search_logged_models(
        self,
        experiment_ids: list[str],
        filter_string: str | None = None,
        datasets: list[dict[str, Any]] | None = None,
        max_results: int | None = None,
        order_by: list[dict[str, Any]] | None = None,
        page_token: str | None = None,
    ) -> PagedList[LoggedModel]:
        """
        Search for logged models that match the specified search criteria.

        Args:
            experiment_ids: List of experiment ids to scope the search.
            filter_string: A search filter string.
            datasets: List of dictionaries to specify datasets on which to apply metrics filters.
                The following fields are supported:

                name (str): Required. Name of the dataset.
                digest (str): Optional. Digest of the dataset.
            max_results: Maximum number of logged models desired.
            order_by: List of dictionaries to specify the ordering of the search results.
                The following fields are supported:

                field_name (str): Required. Name of the field to order by, e.g. "metrics.accuracy".
                ascending: (bool): Optional. Whether the order is ascending or not.
                dataset_name: (str): Optional. If ``field_name`` refers to a metric, this field
                    specifies the name of the dataset associated with the metric. Only metrics
                    associated with the specified dataset name will be considered for ordering.
                    This field may only be set if ``field_name`` refers to a metric.
                dataset_digest (str): Optional. If ``field_name`` refers to a metric, this field
                    specifies the digest of the dataset associated with the metric. Only metrics
                    associated with the specified dataset name and digest will be considered for
                    ordering. This field may only be set if ``dataset_name`` is also set.
            page_token: Token specifying the next page of results.

        Returns:
            A :py:class:`PagedList <mlflow.store.entities.PagedList>` of
            :py:class:`LoggedModel <mlflow.entities.LoggedModel>` objects.
        """

        raise NotImplementedError(self.__class__.__name__)

    def finalize_logged_model(self, model_id: str, status: LoggedModelStatus) -> LoggedModel:
        """
        Finalize a model by updating its status.

        Args:
            model_id: ID of the model to finalize.
            status: Final status to set on the model.

        Returns:
            The updated model.
        """
        raise NotImplementedError(self.__class__.__name__)

    def set_logged_model_tags(self, model_id: str, tags: list[LoggedModelTag]) -> None:
        """
        Set tags on the specified logged model.

        Args:
            model_id: ID of the model.
            tags: Tags to set on the model.

        Returns:
            None
        """
        raise NotImplementedError(self.__class__.__name__)

    def set_model_versions_tags(self, name: str, version: str, model_id: str) -> None:
        mvs = [{"name": name, "version": version}]
        model = self.get_logged_model(model_id)
        if existing_mvs := model.tags.get(mlflow_tags.MLFLOW_MODEL_VERSIONS):
            existing_mvs = json.loads(existing_mvs)
            if mvs[0] not in existing_mvs:
                mvs = existing_mvs + mvs

        self.set_logged_model_tags(
            model_id,
            [
                LoggedModelTag(
                    key=mlflow_tags.MLFLOW_MODEL_VERSIONS,
                    value=json.dumps(mvs),
                )
            ],
        )

    def delete_logged_model_tag(self, model_id: str, key: str) -> None:
        """
        Delete a tag from the specified logged model.

        Args:
            model_id: ID of the model.
            key: Key of the tag to delete.
        """
        raise NotImplementedError(self.__class__.__name__)

    def get_logged_model(self, model_id: str) -> LoggedModel:
        """
        Fetch the logged model with the specified ID.

        Args:
            model_id: ID of the model to fetch.

        Returns:
            The fetched model.
        """
        raise NotImplementedError(self.__class__.__name__)

    def delete_logged_model(self, model_id: str) -> None:
        """
        Delete the logged model with the specified ID.

        Args:
            model_id: ID of the model to delete.
        """
        raise NotImplementedError(self.__class__.__name__)

    @abstractmethod
    def link_traces_to_run(self, trace_ids: list[str], run_id: str) -> None:
        """
        Link multiple traces to a run by creating entity associations.

        Args:
            trace_ids: List of trace IDs to link to the run. Maximum 100 traces allowed.
            run_id: ID of the run to link traces to.

        Raises:
            MlflowException: If more than 100 traces are provided.
        """
