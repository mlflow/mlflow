import json
import os
from unittest.mock import MagicMock, patch

import pytest

from mlflow.genai.scorers.builtin_scorers import Completeness

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="MLflow job execution is not supported on Windows"
)


def make_online_scorer_dict(scorer, sample_rate: float = 1.0):
    return {
        "name": scorer.name,
        "experiment_id": "exp1",
        "serialized_scorer": json.dumps(scorer.model_dump()),
        "sample_rate": sample_rate,
        "filter_string": None,
    }


def test_run_online_trace_scorer_job_calls_processor():
    from mlflow.genai.scorers.job import run_online_trace_scorer_job

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


def test_run_online_session_scorer_job_calls_processor():
    from mlflow.genai.scorers.job import run_online_session_scorer_job

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
    from mlflow.genai.scorers.job import run_online_scoring_scheduler
    from mlflow.genai.scorers.online.entities import OnlineScorer

    # Create mock online scorers
    mock_scorer1 = OnlineScorer(
        name="completeness",
        experiment_id="exp1",
        serialized_scorer='{"name": "completeness"}',
        sample_rate=1.0,
        filter_string=None,
    )
    mock_scorer2 = OnlineScorer(
        name="relevance",
        experiment_id="exp2",
        serialized_scorer='{"name": "relevance"}',
        sample_rate=0.5,
        filter_string=None,
    )

    mock_tracking_store = MagicMock()
    mock_tracking_store.get_active_online_scorers.return_value = [mock_scorer1, mock_scorer2]

    with (
        patch("mlflow.genai.scorers.job._get_tracking_store", return_value=mock_tracking_store),
        patch("mlflow.server.jobs.submit_job") as mock_submit_job,
    ):
        run_online_scoring_scheduler()

        # Verify submit_job was called for both experiments (2 experiments x 2 job types = 4 calls)
        assert mock_submit_job.call_count == 4

        # Verify correct job functions and parameters were passed
        from mlflow.genai.scorers.job import (
            run_online_session_scorer_job,
            run_online_trace_scorer_job,
        )

        call_args_list = mock_submit_job.call_args_list
        trace_scorer_calls = [
            call for call in call_args_list if call[0][0] == run_online_trace_scorer_job
        ]
        session_scorer_calls = [
            call for call in call_args_list if call[0][0] == run_online_session_scorer_job
        ]

        assert len(trace_scorer_calls) == 2
        assert len(session_scorer_calls) == 2

        # Verify experiment IDs are present in the calls
        exp_ids_in_calls = {call[0][1]["experiment_id"] for call in call_args_list}
        assert exp_ids_in_calls == {"exp1", "exp2"}
