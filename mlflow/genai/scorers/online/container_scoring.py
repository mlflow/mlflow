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

from mlflow.entities import Trace
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


def container_scoring_enabled() -> bool:
    """Whether online scoring jobs run custom scorers in job containers."""
    return MLFLOW_JOB_CUSTOM_SCORER_EXECUTOR_BACKEND.get() == DOCKER_EXECUTOR_BACKEND


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
        # Online scoring jobs run one at a time per experiment, so this label is used by at most
        # one container at a time, and a container it finds at the start of a job is left over
        # from an earlier job that was ended while it ran.
        self._container_label = f"{job_name}-{experiment_id}"
        # Online scoring jobs are submitted without a timeout, so they get the default one.
        self._job_timeout = MLFLOW_SERVER_JOB_DEFAULT_TIMEOUT.get()
        self._job_deadline = time.monotonic() + self._job_timeout
        self._client = None
        self._setup_error: str | None = None

    def _get_client(self):
        """The Docker client, set up once per job: leftovers removed and the image ready."""
        if self._setup_error is not None:
            raise MlflowException(self._setup_error)
        if self._client is None:
            try:
                client = job_docker_client()
                remove_job_containers(client, self._container_label)
                prepare_job_image(client)
            except Exception as e:
                # Do not retry a failed setup (for example an image pull) for every batch.
                self._setup_error = f"Could not set up job containers: {e!r}"
                raise
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
        on_result: Callable[[OnlineScorer, list[Trace], dict[str, Any]], None],
        on_failure: Callable[[OnlineScorer, list[Trace], str], None],
    ) -> None:
        """Score sessions in job containers without logging, for the caller to log the results.

        Args:
            work: Each session-level custom scorer with the traces of one session to score.
            on_result: Called with a scorer, a session's traces, and the checked results by trace
                ID (each with ``assessments`` as dictionaries and ``failures``).
            on_failure: Called with a scorer, the traces of a session that failed, and the error.
        """
        for online_scorer, traces, trace_results, error in self._run_batches(
            work, log_assessments=False
        ):
            if error is not None:
                on_failure(online_scorer, traces, error)
            else:
                on_result(online_scorer, traces, trace_results)

    def _run_batches(
        self, batches: list[tuple[OnlineScorer, list[Trace]]], log_assessments: bool
    ) -> Iterator[tuple[OnlineScorer, list[Trace], dict[str, Any] | None, str | None]]:
        """Run batches one at a time, yielding each with its results by trace ID or its error."""
        for index, (online_scorer, batch) in enumerate(batches):
            batches_left = len(batches) - index
            time_left = self._job_deadline - time.monotonic()
            timeout = min(
                (time_left - self._job_timeout * _RESERVED_TIME_SHARE) / batches_left,
                self._job_timeout / 2,
            )
            try:
                result = self._run_batch(online_scorer, batch, timeout, log_assessments)
            except Exception as e:
                _logger.warning(
                    f"Failed to run custom scorer '{online_scorer.name}' in a job container",
                    exc_info=True,
                )
                yield online_scorer, batch, None, repr(e)
                continue
            if result.status != JobStatus.SUCCEEDED:
                error = (
                    result.error_message or f"The job container ended with status {result.status}."
                )
                yield online_scorer, batch, None, error
            else:
                yield online_scorer, batch, json.loads(result.result), None

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
