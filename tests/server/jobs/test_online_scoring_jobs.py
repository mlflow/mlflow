import json
import os
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities._job_status import JobStatus
from mlflow.genai.judges import make_judge
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


def make_online_scorer_dict(scorer, sample_rate: float = 1.0):
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
    trace_scorer = Completeness()

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
        serialized_scorer=json.dumps(trace_scorer.model_dump()),
        online_config=config1,
    )
    mock_scorer2 = OnlineScorer(
        name="relevance",
        serialized_scorer=json.dumps(trace_scorer.model_dump()),
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

        # Verify trace job has 2 scorers
        trace_params = trace_scorer_calls[0].args[1]
        assert len(trace_params["online_scorers"]) == 2
        assert trace_params["experiment_id"] == "exp1"

        # Verify session job has 1 scorer
        session_params = session_scorer_calls[0].args[1]
        assert len(session_params["online_scorers"]) == 1
        assert session_params["experiment_id"] == "exp1"
