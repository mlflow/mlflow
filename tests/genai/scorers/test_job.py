from unittest import mock

import pytest

from mlflow.entities import Trace, TraceData
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.genai.evaluation.entities import EvalItem, EvalResult
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.base import SCORER_BACKEND_TRACKING
from mlflow.genai.scorers.job import _run_session_scorer, invoke_scorer_job
from mlflow.tracing.constant import TraceMetadataKey


@pytest.fixture(autouse=True)
def reset_session_level_online_warned(monkeypatch):
    """Isolate the once-per-process session_level warning set between tests."""
    monkeypatch.setattr("mlflow.genai.scorers.base._SESSION_LEVEL_ONLINE_WARNED", set())


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


def test_run_session_scorer_warns_once_for_session_level():
    @scorer(session_level="episode")
    def session_scorer(session):
        return len(session)

    trace = Trace(
        info=TraceInfo(
            trace_id="tr-1",
            trace_location=TraceLocation.from_experiment_id("0"),
            request_time=1000,
            execution_duration=100,
            state=TraceState.OK,
            trace_metadata={TraceMetadataKey.TRACE_SESSION: '["trip-2","ep-2"]'},
            tags={},
        ),
        data=TraceData(spans=[]),
    )
    eval_result = EvalResult(eval_item=EvalItem.from_trace(trace), assessments=[])

    with (
        mock.patch("mlflow.genai.scorers.job._fetch_traces_batch", return_value={"tr-1": trace}),
        mock.patch(
            "mlflow.genai.scorers.job.evaluate_session_level_scorers",
            return_value=eval_result,
        ) as mock_eval,
        mock.patch("mlflow.genai.scorers.base._logger.warning") as mock_warning,
    ):
        _run_session_scorer(session_scorer, ["tr-1"], mock.Mock(), log_assessments=False)
        _run_session_scorer(session_scorer, ["tr-1"], mock.Mock(), log_assessments=False)

    # The scorer itself runs on every invocation; the session_level warning fires once.
    assert mock_eval.call_count == 2
    warnings = [c for c in mock_warning.call_args_list if "session_level" in str(c)]
    assert len(warnings) == 1
    assert "not yet supported for online scoring" in str(warnings[0])
