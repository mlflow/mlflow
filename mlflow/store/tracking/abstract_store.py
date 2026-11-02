import bisect
import json
from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any, Literal

from mlflow.entities import (
    DatasetInput,
    DatasetRecord,
    LifecycleStage,
    LoggedModel,
    LoggedModelInput,
    LoggedModelOutput,
    LoggedModelParameter,
    LoggedModelStatus,
    LoggedModelTag,
    ScorerVersion,
    ViewType,
)

if TYPE_CHECKING:
    from mlflow.entities import EvaluationDataset
    from mlflow.genai.label_schemas.label_schemas import InputType, LabelSchema
    from mlflow.genai.review_queues import ReviewQueue, ReviewQueueItem
    from mlflow.genai.scorers.online.entities import (
        OnlineScorer,
        OnlineScoringConfig,
    )
from mlflow.entities.metric import MetricWithRunId
from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.workspace import TraceArchivalConfig
from mlflow.exceptions import MlflowException, MlflowNotImplementedException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.store.entities.paged_list import PagedList
from mlflow.store.tracking import (
    SEARCH_MAX_RESULTS_DEFAULT,
)
from mlflow.store.tracking.abstract_trace_backend import AbstractTraceBackend
from mlflow.store.tracking.gateway import GatewayStoreMixin
from mlflow.store.tracking.mcp_server_registry import MCPServerRegistryMixin
from mlflow.store.tracking.skill_registry import SkillRegistryMixin
from mlflow.store.workspace.abstract_store import ResolvedTraceArchivalConfig
from mlflow.utils import mlflow_tags
from mlflow.utils.annotations import developer_stable, requires_sql_backend
from mlflow.utils.async_logging.async_logging_queue import AsyncLoggingQueue
from mlflow.utils.async_logging.run_operations import RunOperations


@developer_stable
class AbstractStore(
    SkillRegistryMixin, MCPServerRegistryMixin, GatewayStoreMixin, AbstractTraceBackend
):
    """
    Abstract class for Backend Storage.
    This class defines the API interface for front ends to connect with various types of backends.
    """

    def __init__(self, trace_backend: AbstractTraceBackend | None = None):
        """
        Initialize shared tracking-store state.

        Args:
            trace_backend: Optional backend requiring explicit host support. Concrete hosts
                bind its tracking-store context after their own initialization succeeds.
        """
        if trace_backend is not None and not self.supports_trace_backends():
            raise MlflowException.invalid_parameter_value(
                "This tracking store does not support trace backends."
            )
        self._async_logging_queue = AsyncLoggingQueue(logging_func=self.log_batch)

    @staticmethod
    def supports_trace_backends() -> bool:
        return False

    def disable_trace_backend(self) -> AbstractContextManager[None]:
        """Temporarily use native operations; supporting hosts must preserve context isolation."""
        raise MlflowNotImplementedException()

    @property
    def supports_workspaces(self) -> bool:
        """Return whether workspaces are supported by this tracking store."""
        return False

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

    def resolve_trace_archival_config(
        self,
        *,
        default_trace_archival_location: str,
        default_retention: str,
    ) -> ResolvedTraceArchivalConfig:
        """
        Resolve the effective trace archival configuration for the current store context.

        Single-tenant stores use the broader-scope defaults directly. Workspace-aware stores may
        override this to apply workspace-specific archival settings. Any field left as ``None``
        inherits the broader-scope default in core.

        Returns:
            A ``ResolvedTraceArchivalConfig`` describing the effective archival location,
            retention, and whether additional workspace path scoping should be applied by the
            caller.
        """

        return ResolvedTraceArchivalConfig(
            config=TraceArchivalConfig(
                location=default_trace_archival_location,
                retention=default_retention,
            ),
            append_workspace_prefix=False,
        )

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

    @staticmethod
    def _evenly_spaced_indices(total: int, max_results: int) -> list[int]:
        """Return at most ``max_results`` evenly spaced indices into a sequence of ``total`` items,
        always including the first and last index.
        """
        if total <= 0:
            return []
        if total <= max_results:
            return list(range(total))
        if max_results == 1:
            return [total - 1]
        step = (total - 1) / (max_results - 1)
        return sorted({round(i * step) for i in range(max_results)})

    def get_metric_history_bulk_interval(
        self,
        run_ids: list[str],
        metric_key: str,
        max_results: int,
        start_step: int | None,
        end_step: int | None,
    ) -> list[MetricWithRunId]:
        """
        Return a list of metric objects for a given metric across multiple runs,
        sampled within a specified step range.

        This method collects metric history from multiple runs, samples the steps
        to limit the result size, and returns metrics for the sampled steps. The
        sampling preserves min/max steps to maintain data boundaries.

        Each run is sampled independently, so runs with differing history lengths may return
        different steps. Consumers should align runs by step value rather than by index.

        Args:
            run_ids: List of unique identifiers for runs.
            metric_key: Metric name to retrieve across runs.
            max_results: Maximum number of steps to sample from the step range.
            start_step: Starting step of the range (inclusive). Must be provided together with
                end_step; if both are None the full range is used (starting from 0).
            end_step: Ending step of the range (inclusive). Must be provided together with
                start_step; if both are None the full range is used (up to the maximum step).

        Returns:
            A list of `MetricWithRunId` objects containing metric data for the sampled
            steps across all specified runs.
        """

        # get a list of all steps for all runs. this is necessary
        # because we can't assume that every step was logged, so
        # sampling needs to be done on the steps that actually exist
        if (start_step is None) != (end_step is None):
            raise MlflowException.invalid_parameter_value(
                "Both start_step and end_step must be provided together, "
                "or neither should be provided."
            )
        max_results = max(1, max_results)
        all_runs = [
            [m.step for m in self.get_metric_history(run_id, metric_key)] for run_id in run_ids
        ]

        # save mins and maxes to be added back later
        all_mins_and_maxes = {step for run in all_runs if run for step in [min(run), max(run)]}
        all_steps = sorted({step for sublist in all_runs for step in sublist})

        # init start and end step if not provided in args
        if start_step is None and end_step is None:
            start_step = 0
            end_step = all_steps[-1] if all_steps else 0

        # remove any steps outside of the range
        all_mins_and_maxes = {step for step in all_mins_and_maxes if start_step <= step <= end_step}

        # doing extra iterations here shouldn't badly affect performance,
        # since the number of steps at this point should be relatively small
        # (MAX_RESULTS_PER_RUN + len(all_mins_and_maxes))

        start_idx = bisect.bisect_left(all_steps, start_step)
        end_idx = bisect.bisect_right(all_steps, end_step)
        if end_idx - start_idx <= max_results:
            sampled_steps = set(all_steps[start_idx:end_idx])
        else:
            num_steps = end_idx - start_idx
            interval = num_steps / max_results
            sampled_steps = set()

            for i in range(0, max_results):
                idx = start_idx + int(i * interval)
                if idx < end_idx:
                    sampled_steps.add(all_steps[idx])

            sampled_steps.add(all_steps[end_idx - 1])

        steps = sorted(sampled_steps.union(all_mins_and_maxes))
        # Bound the rows returned per run to roughly the requested sample size. ``steps`` already
        # holds at most ~``max_results`` distinct steps, so ``len(steps)`` is the expected result
        # size when one value is logged per step. Capping here keeps the response bounded when many
        # values share a single step (e.g. the default ``step=0``), which would otherwise return
        # up to ``MAX_RESULTS_GET_METRIC_HISTORY`` rows for that step and cause large memory spikes.
        # The rows are sampled evenly across the run's full ordered history rather than truncated,
        # so steps at the end of the range are not crowded out by steps holding many values.
        per_run_max_results = max(max_results, len(steps))
        step_set = set(steps)
        metrics_with_run_ids = []
        for run_id in run_ids:
            run_metrics = sorted(
                (m for m in self.get_metric_history(run_id, metric_key) if m.step in step_set),
                key=lambda metric: (metric.step, metric.timestamp),
            )
            metrics_with_run_ids.extend(
                MetricWithRunId(run_id=run_id, metric=run_metrics[i])
                for i in self._evenly_spaced_indices(len(run_metrics), per_run_max_results)
            )
        return metrics_with_run_ids

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

    def get_logged_model(self, model_id: str, allow_deleted: bool = False) -> LoggedModel:
        """
        Fetch the logged model with the specified ID.

        Args:
            model_id: ID of the model to fetch.
            allow_deleted: If ``True``, allow fetching logged models in the deleted lifecycle
                stage. Defaults to ``False``.

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

    @requires_sql_backend
    def create_dataset(
        self,
        name: str,
        tags: dict[str, str] | None = None,
        experiment_ids: list[str] | None = None,
    ) -> "EvaluationDataset":
        """
        Create a new evaluation dataset.

        Args:
            name: The name of the evaluation dataset.
            tags: Optional tags to associate with the dataset.
            experiment_ids: List of experiment IDs to associate with the dataset.

        Returns:
            The created evaluation dataset with populated metadata.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def get_dataset(self, dataset_id: str) -> "EvaluationDataset":
        """
        Get an evaluation dataset by ID.

        Args:
            dataset_id: The ID of the dataset to retrieve.

        Returns:
            The evaluation dataset object.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def delete_dataset(self, dataset_id: str) -> None:
        """
        Delete a dataset and all its records.

        Args:
            dataset_id: The ID of the dataset to delete.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def search_datasets(
        self,
        experiment_ids: list[str] | None = None,
        filter_string: str | None = None,
        max_results: int = 1000,
        order_by: list[str] | None = None,
        page_token: str | None = None,
    ) -> PagedList["EvaluationDataset"]:
        """
        Search for evaluation datasets.

        Args:
            experiment_ids: List of experiment IDs to filter by.
            filter_string: Filter string for dataset names.
            max_results: Maximum number of results to return.
            order_by: Ordering criteria.
            page_token: Token for retrieving the next page of results.

        Returns:
            A PagedList of evaluation datasets.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def upsert_dataset_records(
        self,
        dataset_id: str,
        records: list[dict[str, Any]],
    ) -> dict[str, int]:
        """
        Upsert records into an evaluation dataset.

        Args:
            dataset_id: The ID of the dataset to update.
            records: List of record dictionaries to upsert.

        Returns:
            Dictionary with 'inserted' and 'updated' counts.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def delete_dataset_records(
        self,
        dataset_id: str,
        dataset_record_ids: list[str],
    ) -> int:
        """
        Delete records from an evaluation dataset.

        Args:
            dataset_id: The ID of the dataset.
            dataset_record_ids: List of record IDs to delete.

        Returns:
            The number of records deleted.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def set_dataset_tags(self, dataset_id: str, tags: dict[str, Any]) -> None:
        """
        Set tags for an evaluation dataset.

        This implements an upsert operation - existing tags are merged with new tags.

        Args:
            dataset_id: The ID of the dataset to update.
            tags: Dictionary of tags to update.

        Raises:
            MlflowException: If dataset not found or invalid parameters.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def delete_dataset_tag(self, dataset_id: str, key: str) -> None:
        """
        Delete a tag from an evaluation dataset.

        Args:
            dataset_id: The ID of the dataset.
            key: The tag key to delete.

        Raises:
            MlflowException: If dataset not found.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def get_dataset_experiment_ids(self, dataset_id: str) -> list[str]:
        """
        Get experiment IDs associated with an evaluation dataset.

        This method is used for lazy loading of experiment_ids in the EvaluationDataset entity.

        Args:
            dataset_id: The ID of the dataset.

        Returns:
            List of experiment IDs associated with the dataset.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def _load_dataset_records(
        self,
        dataset_id: str,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> tuple[list[DatasetRecord], str | None]:
        """
        Load dataset records with pagination support.

        This method is used by handlers and for lazy loading of records in the
        EvaluationDataset entity.

        Args:
            dataset_id: The ID of the dataset.
            max_results: Maximum number of records to return. If None, returns all records.
            page_token: Token for pagination. If None, starts from the beginning.

        Returns:
            Tuple of (list of DatasetRecord objects, next_page_token).
            next_page_token is None if there are no more records.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def add_dataset_to_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> "EvaluationDataset":
        """
        Add a dataset to additional experiments.

        Args:
            dataset_id: The ID of the dataset to update
            experiment_ids: List of experiment IDs to associate with the dataset

        Returns:
            The updated EvaluationDataset
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def remove_dataset_from_experiments(
        self, dataset_id: str, experiment_ids: list[str]
    ) -> "EvaluationDataset":
        """
        Remove a dataset from experiments.

        Args:
            dataset_id: The ID of the dataset to update
            experiment_ids: List of experiment IDs to disassociate from the dataset

        Returns:
            The updated EvaluationDataset

        Note:
            This operation is idempotent - removing non-existent associations
            will not raise an error.
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

    def unlink_traces_from_run(self, trace_ids: list[str], run_id: str) -> None:
        """
        Unlink multiple traces from a run by removing entity associations.

        Args:
            trace_ids: List of trace IDs to unlink from the run.
            run_id: ID of the run to unlink traces from.

        Raises:
            MlflowException: If the operation is not supported or fails.
        """
        raise NotImplementedError(
            f"Unlinking traces from runs is not implemented for {self.__class__.__name__}."
        )

    def link_prompts_to_trace(self, trace_id: str, prompt_versions: list[PromptVersion]) -> None:
        """
        Link multiple prompt versions to a trace by creating entity associations.

        Args:
            trace_id: ID of the trace to link prompt versions to.
            prompt_versions: List of PromptVersion objects to link.

        Raises:
            NotImplementedError: If the operation is not supported by this store.
        """
        raise NotImplementedError(
            f"Linking prompts to traces is not implemented for {self.__class__.__name__}."
        )

    def register_scorer(
        self, experiment_id: str, name: str, serialized_scorer: str
    ) -> ScorerVersion:
        """
        Register a scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.
            serialized_scorer: The serialized scorer string (JSON).

        Returns:
            mlflow.entities.ScorerVersion: The newly registered scorer version with scorer_id.
        """
        raise NotImplementedError(self.__class__.__name__)

    def list_scorers(self, experiment_id) -> list[ScorerVersion]:
        """
        List all scorers for an experiment.

        Args:
            experiment_id: The experiment ID.

        Returns:
            List of mlflow.entities.scorer.ScorerVersion objects
            (latest version for each scorer name).
        """
        raise NotImplementedError(self.__class__.__name__)

    def list_scorers_across_experiments(self, experiment_ids: list[str]) -> list[ScorerVersion]:
        """
        List all scorers across multiple experiments in one batch. The default
        impl just iterates ``list_scorers`` per experiment; ``SqlAlchemyStore``
        overrides with a single JOIN for admin pickers that need to enumerate
        scorers across hundreds of experiments without N+1 round trips.
        """
        result: list[ScorerVersion] = []
        for exp_id in experiment_ids:
            result.extend(self.list_scorers(exp_id))
        return result

    def filter_active_experiment_ids(self, experiment_ids: list[str]) -> list[str]:
        """
        Given a bounded, caller-supplied batch of experiment IDs, return the
        subset that exist and are ACTIVE. This is NOT a general-purpose search:
        the caller must already know exactly which IDs it's asking about, and
        the result size is capped by the input size. For open-ended enumeration
        (e.g. "all active experiments in a workspace", where the result size is
        unknown ahead of time), use ``search_experiments`` instead.

        The default impl checks each ID individually via ``get_experiment``;
        ``SqlAlchemyStore`` overrides with a chunked batch query.
        """
        result: list[str] = []
        for exp_id in experiment_ids:
            try:
                experiment = self.get_experiment(exp_id)
            except MlflowException as exc:
                if exc.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
                    raise
                continue
            if experiment.lifecycle_stage == LifecycleStage.ACTIVE:
                result.append(exp_id)
        return result

    def get_scorer(self, experiment_id, name, version=None) -> ScorerVersion:
        """
        Get a specific scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.
            version: The scorer version. If None, returns the scorer with maximum version.

        Returns:
            A ScorerVersion entity object.

        Raises:
            MlflowException: If scorer is not found.
        """
        raise NotImplementedError(self.__class__.__name__)

    def list_scorer_versions(self, experiment_id, name) -> list[ScorerVersion]:
        """
        List all versions of a specific scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.

        Returns:
            List of mlflow.entities.scorer.ScorerVersion objects for all versions of the scorer.

        Raises:
            MlflowException: If scorer is not found.
        """
        raise NotImplementedError(self.__class__.__name__)

    def delete_scorer(self, experiment_id, name, version=None) -> None:
        """
        Delete all versions of a scorer for an experiment.

        Args:
            experiment_id: The experiment ID.
            name: The scorer name.
            version: The scorer version. If None, delete all versions.

        Raises:
            MlflowException: If scorer is not found.
        """
        raise NotImplementedError(self.__class__.__name__)

    def upsert_online_scoring_config(
        self,
        experiment_id: str,
        scorer_name: str,
        sample_rate: float,
        filter_string: str | None = None,
    ) -> "OnlineScoringConfig":
        """
        Create or update online scoring configuration for a scorer.

        Args:
            experiment_id: The ID of the Experiment containing the scorer.
            scorer_name: The scorer name.
            sample_rate: The sampling rate (0.0 to 1.0).
            filter_string: Optional filter expression for trace selection.

        Returns:
            The created or updated OnlineScoringConfig object.

        Raises:
            MlflowException: If scorer is not found.
        """
        raise NotImplementedError(self.__class__.__name__)

    def get_online_scoring_configs(self, scorer_ids: list[str]) -> list["OnlineScoringConfig"]:
        """
        Get online scoring configurations for multiple scorers by their IDs.

        A single scorer can have multiple configurations (e.g., running in different
        experiments or with different filter strings).

        Args:
            scorer_ids: List of scorer IDs to fetch configurations for.

        Returns:
            A list of OnlineScoringConfig objects for the specified scorers.
            Scorers without configurations are not included.
        """
        raise NotImplementedError(self.__class__.__name__)

    def get_active_online_scorers(self) -> list["OnlineScorer"]:
        """
        Get all active online scorers across all experiments.

        Active online scorers are those with a sample_rate greater than zero.
        Each OnlineScorer contains the serialized scorer and sampling configuration.

        Returns:
            List of OnlineScorer entities with serialized_scorer, sample_rate,
            and filter_string fields populated.
        """
        raise NotImplementedError(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Label schemas: UI rendering hints attached to an experiment.
    # See mlflow/genai/label_schemas/ for the entity dataclasses + validation.
    # ------------------------------------------------------------------

    @requires_sql_backend
    def create_label_schema(
        self,
        experiment_id: str,
        *,
        name: str,
        type: Literal["feedback", "expectation"],
        input: "InputType",
        instruction: str | None = None,
        enable_comment: bool = False,
    ) -> "LabelSchema":
        """Create a new label schema scoped to an experiment.

        Args:
            experiment_id: Parent experiment ID.
            name: Schema name, unique within the experiment. Shown to
                reviewers as the label prompt and used as the assessment key.
            type: Schema type (``"feedback"`` or ``"expectation"``).
            input: One of ``InputPassFail`` / ``InputCategorical`` /
                ``InputNumeric`` / ``InputText``.
            instruction: Optional supplementary guidance shown to reviewers.
            enable_comment: UI hint; persisted but not consumed server-side.

        Returns:
            The created :py:class:`LabelSchema` with backend-generated
            ``schema_id`` and audit fields.

        Raises:
            MlflowException(INVALID_PARAMETER_VALUE): on validation failure.
            MlflowException(RESOURCE_ALREADY_EXISTS): on
                ``(experiment_id, name)`` collision.
            MlflowException(RESOURCE_DOES_NOT_EXIST): if the experiment
                doesn't exist.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def get_label_schema(self, schema_id: str) -> "LabelSchema":
        """Fetch a label schema by its server-generated ID.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if no schema has the given ID.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def get_label_schema_by_name(self, experiment_id: str, name: str) -> "LabelSchema":
        """Fetch a label schema by ``(experiment_id, name)``.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if no schema matches.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def list_label_schemas(
        self,
        experiment_id: str,
        max_results: int = 100,
        page_token: str | None = None,
    ) -> PagedList["LabelSchema"]:
        """List label schemas for an experiment, ordered by creation time descending."""
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def update_label_schema(
        self,
        schema_id: str,
        *,
        name: str | None = None,
        instruction: str | None = None,
        enable_comment: bool | None = None,
        input: "InputType | None" = None,
    ) -> "LabelSchema":
        """Sparse update on a label schema.

        ``type`` is immutable post-create and is not accepted as a parameter.
        Renames are supported (set ``name`` to the new value); existing
        assessment rows are not migrated.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if schema_id doesn't exist.
            MlflowException(INVALID_PARAMETER_VALUE): on validation failure.
            MlflowException(RESOURCE_ALREADY_EXISTS): on rename collision.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def delete_label_schema(self, schema_id: str) -> None:
        """Hard-delete a label schema. No-op if it doesn't exist.

        Assessments whose ``name`` matches this schema retain their data
        but render as free-form values in the UI after deletion.
        """
        raise NotImplementedError(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Review queues: named bundles of attached items + questions (label
    # schemas) + assigned users, scoped to an experiment. See
    # mlflow/genai/review_queues/ for the entity dataclasses + validation.
    # ------------------------------------------------------------------

    @requires_sql_backend
    def create_review_queue(
        self,
        experiment_id: str,
        *,
        name: str,
        queue_type: Literal["user", "custom"],
        created_by: str | None = None,
        users: list[str] | None = None,
        schema_ids: list[str] | None = None,
    ) -> "ReviewQueue":
        """Create a review queue scoped to an experiment.

        Args:
            experiment_id: Parent experiment ID.
            name: Queue name, unique within the experiment. For a user
                queue this is the user identifier; ``"default"``/``"Default"``
                are reserved and rejected for custom queues.
            queue_type: ``"user"`` (exactly one assigned user equal to
                ``name``, inherits all of the experiment's schemas) or
                ``"custom"`` (0..N users, an explicit schema subset).
            created_by: User who created the queue (audit).
            users: Assigned users. For a user queue this must be ``[name]``
                or omitted (derived); for a custom queue, 0..N users.
            schema_ids: Attached label-schema ids. Must be empty/omitted for
                a user queue (resolves to all schemas at read time); the
                chosen subset for a custom queue.

        Returns:
            The created :py:class:`ReviewQueue` with backend-generated
            ``queue_id`` and its hydrated ``users`` / ``schema_ids``.

        Raises:
            MlflowException(INVALID_PARAMETER_VALUE): on validation failure.
            MlflowException(RESOURCE_ALREADY_EXISTS): on
                ``(experiment_id, name)`` collision.
            MlflowException(RESOURCE_DOES_NOT_EXIST): if the experiment
                doesn't exist.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def get_or_create_user_queue(
        self,
        experiment_id: str,
        *,
        user: str,
    ) -> "ReviewQueue":
        """Return the user's personal queue for an experiment, creating it
        if absent.

        The queue is owned by its user (``created_by`` is set to ``user``),
        so there is no caller-supplied owner. Atomic and idempotent:
        concurrent callers converge on the single ``(experiment_id, name=user)``
        user queue. This is the backbone of "assign these traces to this
        person" — get-or-create then attach.

        Raises:
            MlflowException(INVALID_PARAMETER_VALUE): on validation failure.
            MlflowException(RESOURCE_DOES_NOT_EXIST): if the experiment
                doesn't exist.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def get_review_queue(self, queue_id: str) -> "ReviewQueue":
        """Fetch a review queue by its server-generated ID.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if no queue has the given ID.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def get_review_queue_by_name(self, experiment_id: str, *, name: str) -> "ReviewQueue":
        """Fetch a review queue by ``(experiment_id, name)``.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if no queue matches.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def list_review_queues(
        self,
        experiment_id: str,
        *,
        user: str | None = None,
        item_id: str | None = None,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList["ReviewQueue"]:
        """List an experiment's review queues, newest first.

        Args:
            experiment_id: Parent experiment ID.
            user: If set, restrict to queues this user is assigned to (their
                user queue plus any custom queue they belong to).
            item_id: If set, restrict to queues that already contain this item
                (a trace id), so callers can see which queues it is a member of.
            max_results: Page size.
            page_token: Opaque continuation token from a previous call.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def update_review_queue(
        self,
        queue_id: str,
        *,
        users: list[str] | None = None,
        schema_ids: list[str] | None = None,
        name: str | None = None,
        new_owner: str | None = None,
    ) -> "ReviewQueue":
        """Edit a custom queue's name, assigned users, attached schemas, and/or owner.

        ``None`` leaves that field untouched; a value (a list, possibly empty,
        for the association sets; a string for ``name`` / ``new_owner``) replaces
        it wholesale. ``queue_type`` is immutable and user queues reject this
        call (their name, user, schemas, and owner are fixed). ``name`` is
        validated as a custom queue name (non-empty, non-reserved) and must be
        unique within the experiment. ``new_owner`` reassigns ownership
        (``created_by``); it is stored case-preserved (matching is
        case-insensitive). Authorization differs per field — owner reassignment
        is MANAGE-only while the other edits are allowed to an owning EDIT user —
        but that gate lives at the handler layer, not here.

        ``schema_ids`` (the questions) are frozen once the queue has any
        attached items: changing them after reviewers start would strand
        answers or leave completed items with never-seen questions. Detach
        the items first to edit questions. Assigned users, name, and owner stay
        editable.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if the queue doesn't exist.
            MlflowException(RESOURCE_ALREADY_EXISTS): if ``name`` collides with
                another queue in the experiment.
            MlflowException(INVALID_PARAMETER_VALUE): on validation failure,
                when called against a user queue, or when changing
                ``schema_ids`` on a queue that already has items.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def delete_review_queue(self, queue_id: str) -> None:
        """Hard-delete a queue and its user / item / schema associations.

        No-op if the queue doesn't exist. Assessments written by reviewers
        against the queue's items are unaffected.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def add_items_to_review_queue(
        self,
        queue_id: str,
        *,
        item_ids: list[str],
        item_type: Literal["trace"] = "trace",
    ) -> list["ReviewQueueItem"]:
        """Attach items to a queue, returning the resulting queue items.

        Idempotent per item: re-attaching an already-attached item is a
        no-op that preserves its existing status. Newly-attached items
        start ``pending``. The returned list covers every requested
        ``item_id`` (newly-added and already-present), in the requested
        order, except that an item removed concurrently (between its insert
        and the read-back) is omitted — callers should reconcile the
        returned items against their requested ``item_ids`` rather than
        assume a 1:1 result.

        ``item_ids`` are stored as soft references — this store method
        validates only their shape, NOT that each is a real trace in the
        queue's experiment/workspace. Trace existence and workspace
        membership are enforced at the handler/API layer that fronts this
        method (the proto/REST stack), keeping attachment resilient to
        archived traces.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if the queue doesn't exist.
            MlflowException(INVALID_PARAMETER_VALUE): on validation failure.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def remove_items_from_review_queue(self, queue_id: str, *, item_ids: list[str]) -> None:
        """Detach items from a queue. No-op for items not attached.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if the queue doesn't exist.
            MlflowException(INVALID_PARAMETER_VALUE): on validation failure.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def list_review_queue_items(
        self,
        queue_id: str,
        *,
        status: Literal["pending", "complete", "declined"] | None = None,
        max_results: int | None = None,
        page_token: str | None = None,
    ) -> PagedList["ReviewQueueItem"]:
        """List a queue's attached items, newest-attached first.

        Args:
            queue_id: The queue to list.
            status: Optional filter on shared-pool status.
            max_results: Page size.
            page_token: Opaque continuation token from a previous call.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if the queue doesn't exist.
        """
        raise NotImplementedError(self.__class__.__name__)

    @requires_sql_backend
    def set_review_queue_item_status(
        self,
        queue_id: str,
        *,
        item_id: str,
        status: Literal["pending", "complete", "declined"],
        completed_by: str | None = None,
    ) -> "ReviewQueueItem":
        """Set the shared-pool status of an attached item.

        Moving to ``complete`` / ``declined`` records ``completed_by`` and a
        completion timestamp; moving back to ``pending`` (reopen) clears
        both. ``completed_by`` is required for the terminal states and
        rejected for ``pending``. This is an explicit reviewer action and is
        never triggered by writing an assessment.

        Raises:
            MlflowException(RESOURCE_DOES_NOT_EXIST): if the queue or the
                attached item doesn't exist.
            MlflowException(INVALID_PARAMETER_VALUE): on validation failure.
        """
        raise NotImplementedError(self.__class__.__name__)
