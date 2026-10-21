import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers.base import Scorer
from mlflow.genai.scorers.builtin_scorers import Completeness, RelevanceToQuery
from mlflow.genai.scorers.job import (
    run_online_scoring_scheduler,
    run_online_session_scorer_job,
    run_online_trace_scorer_job,
)
from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig
from mlflow.server.jobs import get_job, submit_job

from tests.server.jobs.helpers import _setup_job_runner, wait_job_finalize

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


def make_online_scorer_dict(scorer: Scorer, sample_rate: float = 1.0) -> dict[str, Any]:
    return {
        "name": scorer.name,
        "serialized_scorer": json.dumps(scorer.model_dump()),
        "online_config": {
            "online_scoring_config_id": uuid.uuid4().hex,
            "scorer_id": uuid.uuid4().hex,
            "sample_rate": sample_rate,
            "experiment_id": "exp1",
            "filter_string": None,
        },
    }


def test_run_online_trace_scorer_job_calls_processor():
    mock_processor = MagicMock()
    mock_tracking_store = MagicMock()

    with (
        patch("mlflow.genai.scorers.job._get_tracking_store", return_value=mock_tracking_store),
        patch(
            "mlflow.genai.scorers.online.trace_processor.OnlineTraceScoringProcessor.create",
            return_value=mock_processor,
        ) as mock_create,
    ):
        online_scorers = [make_online_scorer_dict(Completeness())]
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


def test_run_online_session_scorer_job_calls_processor():
    mock_processor = MagicMock()
    mock_tracking_store = MagicMock()

    with (
        patch("mlflow.genai.scorers.job._get_tracking_store", return_value=mock_tracking_store),
        patch(
            "mlflow.genai.scorers.online.session_processor.OnlineSessionScoringProcessor.create",
            return_value=mock_processor,
        ) as mock_create,
    ):
        online_scorers = [make_online_scorer_dict(Completeness())]
        run_online_session_scorer_job(experiment_id="exp1", online_scorers=online_scorers)

        exp_id, scorers, store = mock_create.call_args[0]
        assert exp_id == "exp1"
        assert len(scorers) == 1
        assert scorers[0].name == "completeness"
        assert store is mock_tracking_store
        mock_processor.process_sessions.assert_called_once()


def test_scheduler_submits_jobs_via_submit_job():
    # Create trace-level scorers (2)
    trace_scorer_1 = Completeness()
    trace_scorer_2 = RelevanceToQuery()

    # Create session-level scorer (1) using make_judge with {{ conversation }}
    session_scorer = make_judge(
        name="conversation_judge",
        instructions="Evaluate {{ conversation }} for quality",
        feedback_value_type=str,
        model="openai:/gpt-4",
    )

    config1 = OnlineScoringConfig(
        online_scoring_config_id=uuid.uuid4().hex,
        scorer_id=uuid.uuid4().hex,
        sample_rate=1.0,
        experiment_id="exp1",
        filter_string=None,
    )
    config2 = OnlineScoringConfig(
        online_scoring_config_id=uuid.uuid4().hex,
        scorer_id=uuid.uuid4().hex,
        sample_rate=1.0,
        experiment_id="exp1",
        filter_string=None,
    )
    config3 = OnlineScoringConfig(
        online_scoring_config_id=uuid.uuid4().hex,
        scorer_id=uuid.uuid4().hex,
        sample_rate=1.0,
        experiment_id="exp1",
        filter_string=None,
    )

    mock_scorer1 = OnlineScorer(
        name="completeness",
        serialized_scorer=json.dumps(trace_scorer_1.model_dump()),
        online_config=config1,
    )
    mock_scorer2 = OnlineScorer(
        name="relevance_to_query",
        serialized_scorer=json.dumps(trace_scorer_2.model_dump()),
        online_config=config2,
    )
    mock_scorer3 = OnlineScorer(
        name="conversation_judge",
        serialized_scorer=json.dumps(session_scorer.model_dump()),
        online_config=config3,
    )

    mock_tracking_store = MagicMock()
    mock_tracking_store.get_active_online_scorers.return_value = [
        mock_scorer1,
        mock_scorer2,
        mock_scorer3,
    ]

    with (
        patch("mlflow.genai.scorers.job._get_tracking_store", return_value=mock_tracking_store),
        patch("mlflow.genai.scorers.job.submit_job") as mock_submit_job,
    ):
        run_online_scoring_scheduler()

        # Should submit both trace and session jobs for exp1
        assert mock_submit_job.call_count == 2

        # Verify correct job functions and parameters were passed
        call_args_list = mock_submit_job.call_args_list
        trace_scorer_calls = [
            call for call in call_args_list if call[0][0] == run_online_trace_scorer_job
        ]
        session_scorer_calls = [
            call for call in call_args_list if call[0][0] == run_online_session_scorer_job
        ]

        # Should have 1 trace job and 1 session job
        assert len(trace_scorer_calls) == 1
        assert len(session_scorer_calls) == 1

        # Verify trace job has 2 scorers with correct names
        trace_params = trace_scorer_calls[0].args[1]
        assert len(trace_params["online_scorers"]) == 2
        assert trace_params["experiment_id"] == "exp1"
        trace_scorer_names = {s["name"] for s in trace_params["online_scorers"]}
        assert trace_scorer_names == {"completeness", "relevance_to_query"}

        # Verify session job has 1 scorer with correct name
        session_params = session_scorer_calls[0].args[1]
        assert len(session_params["online_scorers"]) == 1
        assert session_params["experiment_id"] == "exp1"
        assert session_params["online_scorers"][0]["name"] == "conversation_judge"


def test_scheduler_skips_invalid_scorers():
    valid_scorer = OnlineScorer(
        name="completeness",
        serialized_scorer=json.dumps(Completeness().model_dump()),
        online_config=OnlineScoringConfig(
            online_scoring_config_id=uuid.uuid4().hex,
            scorer_id=uuid.uuid4().hex,
            sample_rate=1.0,
            experiment_id="exp1",
            filter_string=None,
        ),
    )
    invalid_scorer = OnlineScorer(
        name="invalid_scorer",
        serialized_scorer='{"bad": "data"}',
        online_config=OnlineScoringConfig(
            online_scoring_config_id=uuid.uuid4().hex,
            scorer_id=uuid.uuid4().hex,
            sample_rate=1.0,
            experiment_id="exp1",
            filter_string=None,
        ),
    )

    mock_store = MagicMock()
    mock_store.get_active_online_scorers.return_value = [valid_scorer, invalid_scorer]

    with (
        patch("mlflow.genai.scorers.job._get_tracking_store", return_value=mock_store),
        patch("mlflow.genai.scorers.job.submit_job") as mock_submit,
        patch("mlflow.genai.scorers.job._logger") as mock_logger,
    ):
        run_online_scoring_scheduler()

        mock_logger.warning.assert_called_once()
        assert "invalid_scorer" in mock_logger.warning.call_args[0][0]
        assert mock_submit.call_count == 1  # Only valid scorer submitted


def test_scheduler_skips_disabled_custom_scorer_without_aborting(monkeypatch):
    # In a server process a disabled custom scorer deserializes as non-executing metadata (no raise
    # at classification), so it must be filtered before submit_job. Otherwise submit_job's
    # custom-scorer rejection would raise and abort the whole scheduling pass, skipping the
    # built-in scorers too.
    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "test-boot")
    monkeypatch.delenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", raising=False)
    monkeypatch.delenv("_MLFLOW_IN_JOB_EXECUTOR", raising=False)

    def _online_scorer(name, serialized):
        return OnlineScorer(
            name=name,
            serialized_scorer=serialized,
            online_config=OnlineScoringConfig(
                online_scoring_config_id=uuid.uuid4().hex,
                scorer_id=uuid.uuid4().hex,
                sample_rate=1.0,
                experiment_id="exp1",
                filter_string=None,
            ),
        )

    custom = _online_scorer(
        "custom",
        json.dumps({
            "name": "custom",
            "call_source": "return len(str(outputs))",
            "call_signature": "(outputs)",
            "original_func_name": "custom",
        }),
    )
    builtin = _online_scorer("completeness", json.dumps(Completeness().model_dump()))

    mock_tracking_store = MagicMock()
    mock_tracking_store.get_active_online_scorers.return_value = [custom, builtin]

    with (
        patch("mlflow.genai.scorers.job._get_tracking_store", return_value=mock_tracking_store),
        patch("mlflow.genai.scorers.job.submit_job") as mock_submit_job,
    ):
        run_online_scoring_scheduler()

    submitted = {
        s["name"] for call in mock_submit_job.call_args_list for s in call.args[1]["online_scorers"]
    }
    assert "completeness" in submitted
    assert "custom" not in submitted


def test_scheduler_routes_enabled_session_level_custom_scorer_to_session_job(monkeypatch):
    # In a server process an enabled custom scorer deserializes as non-executing metadata, which
    # must still carry is_session_level_scorer so the scheduler submits it to the session job.
    monkeypatch.setenv("_MLFLOW_SERVER_BOOT_ID", "test-boot")
    monkeypatch.setenv("MLFLOW_SERVER_ENABLE_CUSTOM_SCORERS", "true")
    monkeypatch.delenv("_MLFLOW_IN_JOB_EXECUTOR", raising=False)

    session_custom = OnlineScorer(
        name="session_custom",
        serialized_scorer=json.dumps({
            "name": "session_custom",
            "is_session_level_scorer": True,
            "call_source": "return len(session)",
            "call_signature": "(session)",
            "original_func_name": "session_custom",
        }),
        online_config=OnlineScoringConfig(
            online_scoring_config_id=uuid.uuid4().hex,
            scorer_id=uuid.uuid4().hex,
            sample_rate=1.0,
            experiment_id="exp1",
            filter_string=None,
        ),
    )
    mock_tracking_store = MagicMock()
    mock_tracking_store.get_active_online_scorers.return_value = [session_custom]

    with (
        patch("mlflow.genai.scorers.job._get_tracking_store", return_value=mock_tracking_store),
        patch("mlflow.genai.scorers.job.submit_job") as mock_submit_job,
    ):
        run_online_scoring_scheduler()

    mock_submit_job.assert_called_once()
    job_fn, params = mock_submit_job.call_args.args
    assert job_fn is run_online_session_scorer_job
    assert [s["name"] for s in params["online_scorers"]] == ["session_custom"]
