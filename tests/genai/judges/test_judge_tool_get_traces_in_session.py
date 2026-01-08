from unittest.mock import MagicMock, patch

import pytest

from mlflow.entities.trace import Trace, TraceData
from mlflow.entities.trace_info import TraceInfo as MlflowTraceInfo
from mlflow.entities.trace_location import TraceLocation
from mlflow.entities.trace_state import TraceState
from mlflow.exceptions import MlflowException
from mlflow.genai.judges.tools.get_traces_in_session import GetTracesInSession
from mlflow.genai.judges.tools.types import JudgeToolTraceInfo
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.tracing.utils import TraceMetadataKey
from mlflow.types.llm import ToolDefinition


def test_get_traces_in_session_tool_name() -> None:
    tool = GetTracesInSession()
    assert tool.name == "_get_traces_in_session"


def test_get_traces_in_session_tool_get_definition() -> None:
    tool = GetTracesInSession()
    definition = tool.get_definition()

    assert isinstance(definition, ToolDefinition)
    assert definition.function.name == "_get_traces_in_session"
    assert "session" in definition.function.description.lower()
    assert "multi-turn evaluation" in definition.function.description.lower()
    assert definition.function.parameters.type == "object"
    assert len(definition.function.parameters.required) == 0
    assert definition.type == "function"

    params = definition.function.parameters.properties
    assert "max_results" in params
    assert "order_by" in params
    assert params["max_results"].type == "integer"
    assert params["order_by"].type == "array"


def create_mock_trace(session_id: str | None = None) -> Trace:
    metadata = {}
    if session_id:
        metadata[TraceMetadataKey.TRACE_SESSION] = session_id

    trace_info = MlflowTraceInfo(
        trace_id="current-trace",
        trace_location=TraceLocation.from_experiment_id("exp-123"),
        request_time=1234567890,
        state=TraceState.OK,
        execution_duration=250,
        trace_metadata=metadata,
    )
    return Trace(info=trace_info, data=TraceData(spans=[]))


def test_get_traces_in_session_tool_invoke_success() -> None:
    with patch(
        "mlflow.genai.judges.tools.get_traces_in_session.SearchTracesTool"
    ) as mock_search_tool_class:
        tool = GetTracesInSession()
        current_trace = create_mock_trace("session-123")

        mock_search_tool = MagicMock()
        mock_result = [
            JudgeToolTraceInfo(
                trace_id="trace-1",
                request_time=1000,
                state=TraceState.OK,
                request="What is machine learning?",
                response="Machine learning is a subset of AI...",
                execution_duration=100,
                assessments=[],
            ),
            JudgeToolTraceInfo(
                trace_id="trace-2",
                request_time=2000,
                state=TraceState.OK,
                request="Can you give an example?",
                response="Sure! A common example is...",
                execution_duration=150,
                assessments=[],
            ),
        ]
        mock_search_tool.invoke.return_value = mock_result
        mock_search_tool_class.return_value = mock_search_tool

        result = tool.invoke(current_trace)

        assert len(result) == 2
        assert all(isinstance(ti, JudgeToolTraceInfo) for ti in result)
        assert result[0].trace_id == "trace-1"
        assert result[0].request == "What is machine learning?"
        assert result[1].trace_id == "trace-2"

        mock_search_tool.invoke.assert_called_once_with(
            trace=current_trace,
            filter_string=(
                f"metadata.`{TraceMetadataKey.TRACE_SESSION}` = 'session-123' "
                "AND trace.timestamp < 1234567890"
            ),
            order_by=None,
            max_results=20,
        )


def test_get_traces_in_session_tool_invoke_custom_parameters() -> None:
    with patch(
        "mlflow.genai.judges.tools.get_traces_in_session.SearchTracesTool"
    ) as mock_search_tool_class:
        tool = GetTracesInSession()
        current_trace = create_mock_trace("session-456")

        mock_search_tool = MagicMock()
        mock_search_tool.invoke.return_value = []
        mock_search_tool_class.return_value = mock_search_tool

        tool.invoke(current_trace, max_results=50, order_by=["timestamp DESC"])

        mock_search_tool.invoke.assert_called_once_with(
            trace=current_trace,
            filter_string=(
                f"metadata.`{TraceMetadataKey.TRACE_SESSION}` = 'session-456' "
                "AND trace.timestamp < 1234567890"
            ),
            order_by=["timestamp DESC"],
            max_results=50,
        )


def test_get_traces_in_session_tool_invoke_no_session_id() -> None:
    tool = GetTracesInSession()
    current_trace = create_mock_trace(session_id=None)

    with pytest.raises(MlflowException, match="No session ID found in trace metadata") as exc_info:
        tool.invoke(current_trace)

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_traces_in_session_tool_invoke_invalid_session_id() -> None:
    tool = GetTracesInSession()
    current_trace = create_mock_trace("session@123!invalid")

    with pytest.raises(MlflowException, match="Invalid session ID format") as exc_info:
        tool.invoke(current_trace)

    assert exc_info.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_get_traces_in_session_tool_invoke_empty_result() -> None:
    with patch(
        "mlflow.genai.judges.tools.get_traces_in_session.SearchTracesTool"
    ) as mock_search_tool_class:
        tool = GetTracesInSession()
        current_trace = create_mock_trace("session-123")

        mock_search_tool = MagicMock()
        mock_search_tool.invoke.return_value = []
        mock_search_tool_class.return_value = mock_search_tool

        result = tool.invoke(current_trace)

        assert result == []
        assert len(result) == 0
