from unittest import mock

from mlflow.genai.scorers.base import SCORER_BACKEND_TRACKING
from mlflow.genai.scorers.job import invoke_scorer_job


def test_invoke_scorer_job_restores_registered_version():
    scorer = mock.MagicMock(is_session_level_scorer=False)

    with (
        mock.patch("mlflow.genai.scorers.job.Scorer.model_validate_json", return_value=scorer),
        mock.patch("mlflow.genai.scorers.job._get_tracking_store"),
        mock.patch("mlflow.genai.scorers.job._run_single_turn_scorer_batch", return_value={}),
    ):
        result = invoke_scorer_job(
            experiment_id="exp-123",
            serialized_scorer='{"name":"registered-judge"}',
            scorer_version=3,
            trace_ids=["trace-1"],
        )

    scorer._set_registration_metadata.assert_called_once_with(
        backend=SCORER_BACKEND_TRACKING,
        experiment_id="exp-123",
        sampling_config=None,
        scorer_version=3,
    )
    assert result == {}
