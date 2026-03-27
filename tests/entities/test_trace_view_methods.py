from unittest import mock

from mlflow.entities.trace import Trace
from mlflow.entities.trace_data import TraceData

from tests.tracing.helper import create_test_trace_info


def _make_trace(trace_id="test-trace-id"):
    return Trace(
        info=create_test_trace_info(trace_id),
        data=TraceData(spans=[]),
    )


def test_create_view_delegates_to_client():
    trace = _make_trace()
    mock_view = mock.MagicMock()
    with mock.patch(
        "mlflow.tracking.MlflowClient.create_trace_view", return_value=mock_view
    ) as mock_create:
        result = trace.create_view(
            name="my-view",
            span_filter="span_type == 'LLM'",
            input_path="$.messages",
            output_path="$.content",
            created_by="user1",
            description="A test view",
        )
        mock_create.assert_called_once_with(
            trace_id="test-trace-id",
            name="my-view",
            span_filter="span_type == 'LLM'",
            input_path="$.messages",
            output_path="$.content",
            created_by="user1",
            description="A test view",
        )
    assert result is mock_view


def test_views_delegates_to_client():
    trace = _make_trace()
    mock_views = [mock.MagicMock(), mock.MagicMock()]
    with mock.patch(
        "mlflow.tracking.MlflowClient.list_trace_views", return_value=mock_views
    ) as mock_list:
        result = trace.views
        mock_list.assert_called_once_with(trace_id="test-trace-id")
    assert result is mock_views


def test_delete_view_delegates_to_client():
    trace = _make_trace()
    with mock.patch("mlflow.tracking.MlflowClient.delete_trace_view") as mock_delete:
        trace.delete_view("tv-123")
        mock_delete.assert_called_once_with(trace_id="test-trace-id", view_id="tv-123")


def test_summarize_calls_invoke_judge_model():
    trace = _make_trace()
    mock_feedback = mock.MagicMock()
    mock_feedback.value = "summary text"
    with mock.patch(
        "mlflow.genai.judges.utils.invocation_utils.invoke_judge_model",
        return_value=mock_feedback,
    ) as mock_invoke:
        result = trace.summarize()
        mock_invoke.assert_called_once_with(
            model_uri="openai:/gpt-4o-mini",
            prompt="Summarize this trace concisely. Focus on what the agent did, key decisions, and the outcome.",
            assessment_name="trace_summary",
            trace=trace,
        )
    assert result == "summary text"


def test_analyze_calls_invoke_judge_model():
    trace = _make_trace()
    mock_feedback = mock.MagicMock()
    mock_feedback.value = "analysis text"
    with mock.patch(
        "mlflow.genai.judges.utils.invocation_utils.invoke_judge_model",
        return_value=mock_feedback,
    ) as mock_invoke:
        result = trace.analyze("What went wrong?")
        mock_invoke.assert_called_once_with(
            model_uri="openai:/gpt-4o-mini",
            prompt="What went wrong?",
            assessment_name="trace_analysis",
            trace=trace,
        )
    assert result == "analysis text"
