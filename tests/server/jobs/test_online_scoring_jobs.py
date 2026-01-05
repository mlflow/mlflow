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
        patch("mlflow.server.handlers._get_tracking_store", return_value=mock_tracking_store),
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
