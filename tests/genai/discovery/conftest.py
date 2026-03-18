import pytest

import mlflow
from mlflow.entities.span import SpanType
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
    return mlflow.get_trace(mlflow.get_last_active_trace_id())


@pytest.fixture
def make_trace():
    def _make(
        request="What is MLflow?",
        response="MLflow is an ML platform.",
        session_id=None,
        error_span=False,
    ):
        return _create_trace(
            request=request,
            response=response,
            session_id=session_id,
            error_span=error_span,
        )

    return _make
