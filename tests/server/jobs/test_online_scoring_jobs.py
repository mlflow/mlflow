import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.genai.scorers.builtin_scorers import Completeness, RelevanceToQuery
from mlflow.genai.scorers.job import run_online_trace_scorer_job
from mlflow.server import (
    ARTIFACT_ROOT_ENV_VAR,
    BACKEND_STORE_URI_ENV_VAR,
    HUEY_STORAGE_PATH_ENV_VAR,
)
from mlflow.server.jobs import (
    _ALLOWED_JOB_NAME_LIST,
    _SUPPORTED_JOB_FUNCTION_LIST,
    get_job,
    submit_job,
)
from mlflow.server.jobs.utils import _launch_job_runner

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


def _get_mlflow_repo_home():
    root = str(Path(__file__).resolve().parents[3])
    return f"{root}{os.pathsep}{path}" if (path := os.environ.get("PYTHONPATH")) else root


@contextmanager
def _launch_job_runner_for_test():
    new_pythonpath = _get_mlflow_repo_home()
    with _launch_job_runner(
        {"PYTHONPATH": new_pythonpath},
        os.getpid(),
    ) as proc:
        try:
            yield proc
        finally:
            proc.kill()


@contextmanager
def _setup_job_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    supported_job_functions: list[str],
    allowed_job_names: list[str],
    backend_store_uri: str | None = None,
):
    backend_store_uri = backend_store_uri or f"sqlite:///{tmp_path / 'mlflow.db'}"
    huey_store_path = tmp_path / "huey_store"
    huey_store_path.mkdir()
    default_artifact_root = str(tmp_path / "artifacts")
    try:
        monkeypatch.setenv("MLFLOW_SERVER_ENABLE_JOB_EXECUTION", "true")
        monkeypatch.setenv(BACKEND_STORE_URI_ENV_VAR, backend_store_uri)
        monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, default_artifact_root)
        monkeypatch.setenv(HUEY_STORAGE_PATH_ENV_VAR, str(huey_store_path))
        monkeypatch.setenv("_MLFLOW_SUPPORTED_JOB_FUNCTION_LIST", ",".join(supported_job_functions))
        monkeypatch.setenv("_MLFLOW_ALLOWED_JOB_NAME_LIST", ",".join(allowed_job_names))
        _SUPPORTED_JOB_FUNCTION_LIST.clear()
        _SUPPORTED_JOB_FUNCTION_LIST.extend(supported_job_functions)
        _ALLOWED_JOB_NAME_LIST.clear()
        _ALLOWED_JOB_NAME_LIST.extend(allowed_job_names)

        with _launch_job_runner_for_test() as job_runner_proc:
            time.sleep(10)
            yield job_runner_proc
    finally:
        # Clear the huey instance cache AFTER killing the runner to ensure clean state for next test
        import mlflow.server.jobs.utils

        mlflow.server.jobs.utils._huey_instance_map.clear()
        if mlflow.server.handlers._job_store is not None:
            # close all db connections and drops connection pool
            mlflow.server.handlers._job_store.engine.dispose()
        mlflow.server.handlers._job_store = None


def wait_job_finalize(job_id, timeout=60):
    beg_time = time.time()
    while time.time() - beg_time <= timeout:
        job = get_job(job_id)
        if JobStatus.is_finalized(job.status):
            return
        time.sleep(0.5)
    raise TimeoutError("The job is not finalized within the timeout.")


def test_run_online_trace_scorer_job_calls_processor():
    from mlflow.genai.scorers.job import run_online_trace_scorer_job
    from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig

    mock_processor = MagicMock()
    mock_tracking_store = MagicMock()

    with (
        patch("mlflow.server.handlers._get_tracking_store", return_value=mock_tracking_store),
        patch(
            "mlflow.genai.scorers.online.trace_processor.OnlineTraceScoringProcessor.create",
            return_value=mock_processor,
        ) as mock_create,
    ):
        # Create OnlineScorer object and convert to dict using asdict()
        scorer = OnlineScorer(
            name="completeness",
            serialized_scorer=json.dumps(Completeness().model_dump()),
            online_config=OnlineScoringConfig(
                online_scoring_config_id="config_id",
                scorer_id="completeness",
                sample_rate=1.0,
                experiment_id="exp1",
                filter_string=None,
            ),
        )
        online_scorers = [asdict(scorer)]
        run_online_trace_scorer_job(experiment_id="exp1", online_scorers=online_scorers)

        exp_id, scorers, store = mock_create.call_args[0]
        assert exp_id == "exp1"
        assert len(scorers) == 1
        assert scorers[0].name == "completeness"
        assert store is mock_tracking_store
        mock_processor.process_traces.assert_called_once()


def test_run_online_trace_scorer_job_runs_exclusively_per_experiment(monkeypatch, tmp_path: Path):
    """
    Test that online trace scorer jobs are exclusive per experiment_id.
    When two jobs are submitted for the same experiment with different scorers,
    only one should run and the other should be canceled due to exclusivity.
    """
    from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig

    with _setup_job_runner(
        monkeypatch,
        tmp_path,
        supported_job_functions=["mlflow.genai.scorers.job.run_online_trace_scorer_job"],
        allowed_job_names=["run_online_trace_scorer"],
    ):
        # Create two different scorer lists for the same experiment using asdict()
        scorer1 = OnlineScorer(
            name="completeness",
            serialized_scorer=json.dumps(Completeness().model_dump()),
            online_config=OnlineScoringConfig(
                online_scoring_config_id="config1",
                scorer_id="completeness",
                sample_rate=1.0,
                experiment_id="exp1",
                filter_string=None,
            ),
        )
        scorer2 = OnlineScorer(
            name="relevance_to_query",
            serialized_scorer=json.dumps(RelevanceToQuery().model_dump()),
            online_config=OnlineScoringConfig(
                online_scoring_config_id="config2",
                scorer_id="relevance_to_query",
                sample_rate=1.0,
                experiment_id="exp1",
                filter_string=None,
            ),
        )

        params1 = {"experiment_id": "exp1", "online_scorers": [asdict(scorer1)]}
        params2 = {"experiment_id": "exp1", "online_scorers": [asdict(scorer2)]}

        # Submit two jobs with same experiment_id but different scorers
        job1_id = submit_job(run_online_trace_scorer_job, params1).job_id
        job2_id = submit_job(run_online_trace_scorer_job, params2).job_id

        wait_job_finalize(job1_id)
        wait_job_finalize(job2_id)

        job1 = get_job(job1_id)
        job2 = get_job(job2_id)

        # One job is canceled (skipped due to exclusive lock on experiment_id),
        # the other either succeeds or fails (we only care about exclusivity, not job success)
        statuses = {job1.status, job2.status}
        assert JobStatus.CANCELED in statuses
        # The non-canceled job should have attempted to run (either SUCCEEDED or FAILED)
        non_canceled_statuses = statuses - {JobStatus.CANCELED}
        assert len(non_canceled_statuses) == 1
        assert non_canceled_statuses.pop() in {JobStatus.SUCCEEDED, JobStatus.FAILED}
