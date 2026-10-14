"""Run the custom ``@scorer`` scorers of an online scoring job in job containers.

When the docker executor backend runs custom scorers (``MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND``
is ``docker``), an online scoring job keeps finding, sampling and logging on the host, but runs its
custom scorers in job containers without network access, one at a time: a batch of traces for a
trace-level scorer, or one session for a session-level scorer. The host writes the traces for the
container, and then checks the container's untrusted result, as for an on-demand scorer run (see
``mlflow.genai.scorers.job._plan_invoke_scorer_job_for_docker``), and logs the assessments.
"""

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from mlflow.entities import Feedback, Trace
from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import (
    MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND,
    MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT,
    MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers.online.entities import OnlineScorer
from mlflow.server.jobs.executor import JobResult
from mlflow.server.jobs.executor_registry import DOCKER_EXECUTOR_BACKEND
from mlflow.server.jobs.job_container import (
    _CONTAINER_INPUT_DIR,
    job_docker_client,
    job_image,
    make_job_container_dirs,
    prepare_job_image,
    remove_job_containers,
    run_plan_in_container,
)

_logger = logging.getLogger(__name__)

# The online job is ended (with its process group, so without any cleanup) at its timeout. Batches
# share the time left minus this fraction of the job timeout, kept in reserve so they and the
# host's logging finish before that happens.
_RESERVED_TIME_SHARE = 0.1

# A failed setup of the job containers (for example while the Docker daemon restarts) is retried
# after each of these delays, in seconds, while the job has time left for it.
_SETUP_RETRY_DELAYS = (5, 15, 30)

# The least time, in seconds, a batch is given (or half the job timeout, if that is less): a job
# container takes some seconds to start and load the scorer before it scores anything. A batch is
# not started once the job has less time left than this.
_MIN_BATCH_TIMEOUT = 30

_OUT_OF_TIME_ERROR = (
    "Not scored: the online scoring job had no time left to run this in a job container. Lower the "
    "sample rate or the number of custom scorers, or raise MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT."
)


def container_scoring_enabled() -> bool:
    """Whether online scoring jobs run custom scorers in job containers."""
    return MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND.get() == DOCKER_EXECUTOR_BACKEND


def _is_lasting_setup_error(e: Exception) -> bool:
    """Whether a failed job container setup would fail again however long the job waits."""
    import docker.errors

    # For example a configured image that does not exist, or a failed build of the default image,
    # which is raised again with the build error as its cause.
    lasting = (docker.errors.NotFound, docker.errors.BuildError)
    return isinstance(e, lasting) or isinstance(e.__cause__, lasting)


class OnlineContainerScorer:
    """Runs one online scoring job's custom scorers in job containers, one batch at a time.

    Args:
        experiment_id: The experiment the job scores.
        job_name: The online scoring job's name. Resource limits for its containers can be set
            with ``MLFLOW_JOB_RESOURCE_LIMITS_<job_name>``.
    """

    def __init__(self, experiment_id: str, job_name: str):
        self._experiment_id = experiment_id
        self._job_name = job_name
        # Online scoring jobs run one at a time per experiment, even across job runners (they hold
        # the experiment's exclusive job lock), so this label is used by at most one container at a
        # time, and a container it finds at the start of a job is left over from an earlier job
        # that was ended while it ran. Only then is it safe to remove: a container left for an
        # experiment that gets no further job stays until removed by hand, for example with
        # ``docker rm -f $(docker ps -aq --filter label=mlflow.job_name=<job_name>)``.
        self._container_label = f"{job_name}-{experiment_id}"
        # Online scoring jobs are submitted without a timeout, so they get the default one.
        self._job_timeout = MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT.get()
        self._job_deadline = time.monotonic() + self._job_timeout
        self._min_batch_timeout = min(_MIN_BATCH_TIMEOUT, self._job_timeout / 2)
        self._client = None
        self._setup_error: str | None = None

    def _time_left(self) -> float:
        """Seconds the job's batches have left, without the reserved time."""
        return self._job_deadline - time.monotonic() - self._job_timeout * _RESERVED_TIME_SHARE

    def _get_client(self):
        """The Docker client, set up once per job: leftovers removed and the image ready."""
        if self._setup_error is not None:
            raise MlflowException(self._setup_error)
        if self._client is None:
            for delay in (*_SETUP_RETRY_DELAYS, None):
                client = None
                try:
                    client = job_docker_client()
                    remove_job_containers(client, self._container_label)
                    prepare_job_image(client)
                    break
                except Exception as e:
                    if client is not None:
                        client.close()
                    if delay is None or delay >= self._time_left() or _is_lasting_setup_error(e):
                        # Remembered, so later batches fail at once instead of setting up again.
                        self._setup_error = f"Could not set up job containers: {e!r}"
                        raise MlflowException(self._setup_error) from e
                    _logger.warning(
                        "Could not set up job containers; retrying in %s seconds: %r", delay, e
                    )
                    time.sleep(delay)
            self._client = client
        return self._client

    def score(
        self,
        work: list[tuple[OnlineScorer, list[Trace]]],
        on_failure: Callable[[OnlineScorer, list[Trace], str], None],
    ) -> None:
        """Score each scorer's traces in job containers and log the assessments.

        Args:
            work: Each custom scorer with the traces it should score.
            on_failure: Called with a scorer, traces that failed, and the error.
        """
        batch_size = MLFLOW_SERVER_SCORER_INVOKE_BATCH_SIZE.get()
        batches = [
            (online_scorer, traces[i : i + batch_size])
            for online_scorer, traces in work
            for i in range(0, len(traces), batch_size)
        ]
        for online_scorer, batch, trace_results, error in self._run_batches(
            batches, log_assessments=True
        ):
            if error is not None:
                on_failure(online_scorer, batch, error)
                continue
            # Report traces the scorer failed on, or that are missing from the result, as on the
            # host.
            for trace in batch:
                trace_result = trace_results.get(trace.info.trace_id)
                if trace_result is None:
                    on_failure(
                        online_scorer,
                        [trace],
                        "The job container returned no result for this trace.",
                    )
                elif trace_result["failures"] and not trace_result["assessments"]:
                    errors = "; ".join(
                        f"{f['error_code']}: {f['error_message']}" for f in trace_result["failures"]
                    )
                    on_failure(online_scorer, [trace], errors)

    def score_sessions(
        self,
        work: list[tuple[OnlineScorer, list[Trace]]],
        on_result: Callable[[OnlineScorer, list[Trace], list[Feedback]], None],
        on_failure: Callable[[OnlineScorer, list[Trace], str], None],
    ) -> None:
        """Score sessions in job containers without logging, for the caller to log the results.

        Args:
            work: Each session-level custom scorer with one session's traces, oldest first.
            on_result: Called with a scorer, a session's traces, and the checked assessments for
                the session's first trace, where session-level results belong.
            on_failure: Called with a scorer, the traces of a session that failed, and the error.
        """
        for online_scorer, traces, trace_results, error in self._run_batches(
            work, log_assessments=False
        ):
            if error is None:
                # Only the first trace's result is used, wherever else the container put results.
                trace_result = trace_results.get(traces[0].info.trace_id)
                if trace_result is None:
                    error = "The job container returned no result for this session."
                elif feedbacks := [
                    Feedback.from_dictionary(a) for a in trace_result["assessments"]
                ]:
                    on_result(online_scorer, traces, feedbacks)
                    continue
                elif trace_result["failures"]:
                    error = "; ".join(
                        f"{f['error_code']}: {f['error_message']}" for f in trace_result["failures"]
                    )
                else:
                    # The scorer returned None, which means no assessment, as on the host.
                    continue
            on_failure(online_scorer, traces, error)

    def _run_batches(
        self, batches: list[tuple[OnlineScorer, list[Trace]]], log_assessments: bool
    ) -> Iterator[tuple[OnlineScorer, list[Trace], dict[str, Any] | None, str | None]]:
        """Run batches one at a time, yielding each with its results by trace ID or its error."""
        for index, (online_scorer, batch) in enumerate(batches):
            try:
                timeout = self._next_batch_timeout(len(batches) - index)
                result = (
                    None
                    if timeout is None
                    else self._run_batch(online_scorer, batch, timeout, log_assessments)
                )
            except Exception as e:
                _logger.warning(
                    f"Failed to run custom scorer '{online_scorer.name}' in a job container",
                    exc_info=True,
                )
                error = e.message if isinstance(e, MlflowException) else repr(e)
                yield online_scorer, batch, None, error
                continue
            if result is None:
                # Setup failed or the job is out of time, so none of the batches left is started.
                if self._setup_error is None:
                    _logger.warning(
                        f"The online scoring job for experiment {self._experiment_id} ran out of "
                        f"time; {len(batches) - index} custom scorer runs were not started."
                    )
                error = self._setup_error or _OUT_OF_TIME_ERROR
                for skipped_scorer, skipped_batch in batches[index:]:
                    yield skipped_scorer, skipped_batch, None, error
                return
            if result.status != JobStatus.SUCCEEDED:
                error = (
                    result.error_message or f"The job container ended with status {result.status}."
                )
                yield online_scorer, batch, None, error
            else:
                yield online_scorer, batch, json.loads(result.result), None

    def _next_batch_timeout(self, batches_left: int) -> float | None:
        """Set up if needed and return the next batch's timeout, or None if it cannot start."""
        if self._setup_error is not None or self._time_left() < self._min_batch_timeout:
            return None
        # Set up first, so the time a slow or retried setup takes is not given to a batch.
        self._get_client()
        time_left = self._time_left()
        if time_left < self._min_batch_timeout:
            return None
        # Batches share the time left, but each gets at least the minimum, so they do not all get a
        # share too short to finish in. Batches that finish early leave their time to later ones.
        return min(max(time_left / batches_left, self._min_batch_timeout), self._job_timeout / 2)

    def _run_batch(
        self,
        online_scorer: OnlineScorer,
        traces: list[Trace],
        timeout: float,
        log_assessments: bool,
    ) -> JobResult:
        # Imported here: mlflow.genai.scorers.job imports the online scoring processors, which
        # import this module.
        from mlflow.genai.scorers.job import _plan_invoke_scorer_job_for_docker

        params = {
            "experiment_id": self._experiment_id,
            "serialized_scorer": online_scorer.serialized_scorer,
            "trace_ids": [trace.info.trace_id for trace in traces],
            "scorer_version": online_scorer.scorer_version,
            "log_assessments": log_assessments,
        }
        client = self._get_client()
        with tempfile.TemporaryDirectory(prefix="mlflow-online-scoring-") as workdir:
            input_dir, output_dir = make_job_container_dirs(Path(workdir))
            plan = _plan_invoke_scorer_job_for_docker(
                params=params, input_dir=input_dir, container_input_dir=_CONTAINER_INPUT_DIR
            )
            return run_plan_in_container(
                client,
                job_image(),
                job_id=self._container_label,
                job_name=self._job_name,
                plan=plan,
                input_dir=input_dir,
                output_dir=output_dir,
                timeout=timeout,
            )
