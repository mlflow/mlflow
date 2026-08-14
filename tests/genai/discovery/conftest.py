import pytest

import mlflow
from mlflow.entities.span import SpanType
from mlflow.entities.trace import Trace
from mlflow.entities.trace_info import TraceInfo
from mlflow.genai.scorers.base import Scorer


class _TestScorer(Scorer):
    """Minimal Scorer subclass for use in discovery pipeline tests."""

    def __call__(self, **kwargs):
        return True


def _create_trace(
    request="What is MLflow?",
    response="MLflow is an ML platform.",
    session_id=None,
    span_type=SpanType.CHAIN,
    error_span=False,
    execution_duration_ms="default",
):
    @mlflow.trace(name="agent", span_type=span_type)
    def _run(question):
        if session_id:
            mlflow.update_current_trace(
                metadata={"mlflow.trace.session": session_id},
            )
        with mlflow.start_span(name="llm_call", span_type=SpanType.LLM) as child:
            child.set_inputs({"prompt": question})
            child.set_outputs({"response": response})
        if error_span:
            with mlflow.start_span(name="tool_call", span_type=SpanType.TOOL) as tool:
                tool.set_inputs({"action": "fetch"})
                tool.record_exception("Connection failed")
        return response

    _run(request)
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    # Override execution_duration if explicitly requested
    if execution_duration_ms != "default":
        trace_info = TraceInfo(
            trace_id=trace.info.trace_id,
            trace_location=trace.info.trace_location,
            request_time=trace.info.timestamp_ms,
            execution_duration=execution_duration_ms,
            state=trace.info.state,
            trace_metadata=trace.info.trace_metadata,
            tags=trace.info.tags,
        )
        trace = Trace(trace_info, trace.data)

    return trace


@pytest.fixture
def make_trace():
    def _make(
        request="What is MLflow?",
        response="MLflow is an ML platform.",
        session_id=None,
        error_span=False,
        execution_duration_ms="default",
    ):
        return _create_trace(
            request=request,
            response=response,
            session_id=session_id,
            error_span=error_span,
            execution_duration_ms=execution_duration_ms,
        )

    return _make
