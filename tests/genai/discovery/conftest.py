import pytest

import mlflow
from mlflow.entities.assessment import Feedback
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.entities.span import SpanType


def _create_trace(
    request_input="What is MLflow?",
    response_output="MLflow is an ML platform.",
    session_id=None,
    span_type=SpanType.CHAIN,
    error_span=False,
):
    @mlflow.trace(name="agent", span_type=span_type)
    def _run(question):
        if session_id:
            mlflow.update_current_trace(
                metadata={"mlflow.trace.session": session_id},
            )
        with mlflow.start_span(name="llm_call", span_type=SpanType.LLM) as child:
            child.set_inputs({"prompt": question})
            child.set_outputs({"response": response_output})
        if error_span:
            with mlflow.start_span(name="tool_call", span_type=SpanType.TOOL) as tool:
                tool.set_inputs({"action": "fetch"})
                tool.record_exception("Connection failed")
        return response_output

    _run(request_input)
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


@pytest.fixture
def make_trace():
    def _make(
        request_input="What is MLflow?",
        response_output="MLflow is an ML platform.",
        session_id=None,
        error_span=False,
    ):
        return _create_trace(
            request_input=request_input,
            response_output=response_output,
            session_id=session_id,
            error_span=error_span,
        )

    return _make


@pytest.fixture
def make_assessment():
    def _make(name, value):
        return Feedback(
            name=name,
            value=value,
            source=AssessmentSource(source_type=AssessmentSourceType.LLM_JUDGE, source_id="test"),
        )

    return _make
